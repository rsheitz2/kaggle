import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from features import build_features, BASELINE_COLS, ENGINEERED_COLS

warnings.filterwarnings("ignore")

EXPERIMENT = "credit-fraud"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
TARGET_COL = "target"   # TODO: update to match actual target column name
ID_COLS: list[str] = [] # TODO: update, e.g. ["id", "loan_id"]


def cv_metrics(model, X, y) -> dict:
    """5-fold stratified CV; threshold-tuned F1 and PR-AUC are the primary metrics."""
    auc_scores, pr_auc_scores, f1_default = [], [], []
    f1_tuned, prec_tuned, rec_tuned, best_thresholds = [], [], [], []

    for tr_idx, val_idx in CV.split(X, y):
        X_tr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_val)[:, 1]

        auc_scores.append(roc_auc_score(y_val, proba))
        pr_auc_scores.append(average_precision_score(y_val, proba))
        f1_default.append(f1_score(y_val, (proba >= 0.5).astype(int), zero_division=0))

        # Sweep thresholds 0.10–0.90 to find the one that maximises F1 on this fold
        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.10, 0.91, 0.02):
            f = f1_score(y_val, (proba >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)

        preds_t = (proba >= best_t).astype(int)
        f1_tuned.append(best_f)
        best_thresholds.append(best_t)
        prec_tuned.append(precision_score(y_val, preds_t, zero_division=0))
        rec_tuned.append(recall_score(y_val, preds_t, zero_division=0))

    return {
        "cv_auc":              float(np.mean(auc_scores)),
        "cv_pr_auc":           float(np.mean(pr_auc_scores)),
        "cv_f1":               float(np.mean(f1_default)),
        "cv_f1_tuned":         float(np.mean(f1_tuned)),
        "cv_precision_tuned":  float(np.mean(prec_tuned)),
        "cv_recall_tuned":     float(np.mean(rec_tuned)),
        "cv_threshold":        float(np.mean(best_thresholds)),
    }


def run(name: str, model, params: dict, X, y, feature_set: str):
    with mlflow.start_run(run_name=f"{name}_{feature_set}"):
        mlflow.set_tag("model", name.split("_")[0])
        mlflow.set_tag("feature_set", feature_set)
        mlflow.set_tag("threshold_strategy", "tuned")
        mlflow.log_params({k: str(v) for k, v in params.items()})

        metrics = cv_metrics(model, X, y)
        mlflow.log_metrics(metrics)
        mlflow.log_param("optimal_threshold", metrics["cv_threshold"])

        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model")

        print(
            f"  {name:38s} [{feature_set:12s}]"
            f"  auc={metrics['cv_auc']:.4f}"
            f"  pr_auc={metrics['cv_pr_auc']:.4f}"
            f"  f1_tuned={metrics['cv_f1_tuned']:.4f}"
            f"  @t={metrics['cv_threshold']:.2f}"
        )
        return metrics["cv_pr_auc"], model


def make_model_grid():
    """Returns list of (display_name, estimator, logged_params_dict)."""
    grid = []

    for C in [0.01, 0.1, 1.0, 10.0]:
        grid.append((
            f"logreg_C{C}",
            make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
            {"C": C, "solver": "lbfgs", "max_iter": 1000},
        ))

    for n_est in [100, 300]:
        for max_d in [None, 10]:
            grid.append((
                f"rf_n{n_est}_d{max_d or 'None'}",
                RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1),
                {"n_estimators": n_est, "max_depth": max_d},
            ))

    for n_est in [100, 200]:
        for max_d in [3, 5]:
            for lr in [0.05, 0.1]:
                grid.append((
                    f"gb_n{n_est}_d{max_d}_lr{lr}",
                    GradientBoostingClassifier(
                        n_estimators=n_est, max_depth=max_d, learning_rate=lr, random_state=42
                    ),
                    {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr},
                ))

    for n_est in [100, 300]:
        for max_d in [3, 6]:
            for lr in [0.05, 0.1]:
                grid.append((
                    f"xgb_n{n_est}_d{max_d}_lr{lr}",
                    XGBClassifier(
                        n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                        eval_metric="logloss", random_state=42, verbosity=0,
                    ),
                    {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr},
                ))

    for n_est in [100, 300]:
        for num_leaves in [31, 63]:
            for lr in [0.05, 0.1]:
                grid.append((
                    f"lgbm_n{n_est}_l{num_leaves}_lr{lr}",
                    LGBMClassifier(
                        n_estimators=n_est, num_leaves=num_leaves, learning_rate=lr,
                        random_state=42, verbose=-1,
                    ),
                    {"n_estimators": n_est, "num_leaves": num_leaves, "learning_rate": lr},
                ))

    return grid


def main():
    train = pd.read_csv("data/train.csv")
    y = train[TARGET_COL].values
    drop_cols = [TARGET_COL] + ID_COLS

    mlflow.set_experiment(EXPERIMENT)
    X_all, feat_cols, enc, scl = build_features(
        train.drop(columns=drop_cols, errors="ignore")
    )

    feat_sets: dict[str, pd.DataFrame] = {}
    if BASELINE_COLS:
        feat_sets["baseline"] = X_all[BASELINE_COLS]
    if ENGINEERED_COLS:
        feat_sets["engineered"] = X_all[ENGINEERED_COLS]
    if not feat_sets:
        feat_sets["all"] = X_all
        print(
            f"NOTE: BASELINE_COLS and ENGINEERED_COLS are empty. "
            f"Training on all {X_all.shape[1]} numeric features.\n"
            f"Fill in features.py after running EDA, then re-run train.py.\n"
        )

    grid = make_model_grid()
    # results[key] = (pr_auc, fitted_model, base_model_name, feature_set_name)
    results: dict[str, tuple] = {}

    for feat_name, X_feat in feat_sets.items():
        print(f"\n=== Feature set: {feat_name} ({X_feat.shape[1]} features) ===")
        for name, model, params in grid:
            pr_auc, fitted = run(name, model, params, X_feat, y, feat_name)
            results[f"{name}__{feat_name}"] = (pr_auc, fitted, name.split("_")[0], feat_name)

    # Voting ensemble: top-3 by cv_pr_auc from the best feature set
    best_feat = "engineered" if "engineered" in feat_sets else list(feat_sets.keys())[-1]
    X_best = feat_sets[best_feat]
    top3 = sorted(
        [(k, v) for k, v in results.items() if v[3] == best_feat],
        key=lambda x: x[1][0],
        reverse=True,
    )[:3]
    estimators = [(f"top{i+1}", clone(v[1])) for i, (_, v) in enumerate(top3)]
    top3_display = [v[2] for _, v in top3]
    print(f"\n=== Voting ensemble (top-3 on {best_feat}: {top3_display}) ===")
    voting = VotingClassifier(estimators=estimators, voting="soft")
    run("voting", voting, {}, X_best, y, best_feat)

    print("\nDone. Run `mlflow ui` to inspect all runs.")


if __name__ == "__main__":
    main()
