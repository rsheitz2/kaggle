# Credit Fraud / First Payment Default

Kaggle competition: `optimizingdefaultmodelbyfirstpaymentdefault`

Binary classification to predict first-payment default (or credit fraud). MLflow tracks every model × hyperparameter × feature-set combination, with per-fold threshold tuning to optimise F1 on imbalanced data.

## Workflow

```
1. Download data       →  kaggle competitions download -c optimizingdefaultmodelbyfirstpaymentdefault -p data/
2. Run EDA             →  open eda.org in Emacs  OR  jupyter lab eda.ipynb
3. Fill in features.py →  update CATEGORICAL_COLS, NUMERIC_COLS, BASELINE_COLS, ENGINEERED_COLS
4. Smoke-test          →  python features.py
5. Train               →  python train.py
6. Browse results      →  mlflow ui  (http://127.0.0.1:5000)
7. Submit              →  python submit.py
```

## Files

| File | Purpose |
|------|---------|
| `eda.org` | Emacs org-babel EDA — interactive Plotly plots saved to `plots/*.html` |
| `eda.ipynb` | Jupyter version of the same EDA (identical content, `fig.show()` inline) |
| `features.py` | `build_features(df, encoder, scaler, fit)` → `(X, cols, encoder, scaler)` |
| `train.py` | Full hyperparameter grid × 2 feature sets; ~65 MLflow runs + voting ensemble |
| `submit.py` | Queries best run by `cv_pr_auc`, applies tuned threshold, writes submission CSV |

## MLflow metrics

| Metric | Description |
|--------|-------------|
| `cv_pr_auc` | PR-AUC (primary — robust to class imbalance) |
| `cv_auc` | ROC-AUC |
| `cv_f1_tuned` | F1 at per-fold optimal threshold |
| `cv_precision_tuned` | Precision at tuned threshold |
| `cv_recall_tuned` | Recall at tuned threshold |
| `optimal_threshold` | Mean threshold that maximised F1 across CV folds |

## Model grid (~32 configs × 2 feature sets)

| Model | Hyperparameters varied |
|-------|----------------------|
| LogisticRegression | C ∈ {0.01, 0.1, 1.0, 10.0} (with StandardScaler pipeline) |
| RandomForest | n_estimators ∈ {100, 300}, max_depth ∈ {None, 10} |
| GradientBoosting | n_estimators ∈ {100, 200}, max_depth ∈ {3, 5}, lr ∈ {0.05, 0.1} |
| XGBoost | n_estimators ∈ {100, 300}, max_depth ∈ {3, 6}, lr ∈ {0.05, 0.1} |
| LightGBM | n_estimators ∈ {100, 300}, num_leaves ∈ {31, 63}, lr ∈ {0.05, 0.1} |

After all runs: soft-voting ensemble from top-3 by `cv_pr_auc`.

## TODO after data download

1. Set `TARGET_COL` in `train.py`, `submit.py`, `eda.org`, `eda.ipynb`
2. Set `ID_COL` / `ID_COLS` in `train.py` and `submit.py`
3. Run EDA → fill in "Key Findings" section
4. Populate `features.py` constants and engineering steps
