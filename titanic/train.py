import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from features import build_features, BASELINE_COLS, REDUCED_COLS

warnings.filterwarnings("ignore")

EXPERIMENT = "titanic"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def cv_metrics(model, X, y):
    acc = cross_val_score(model, X, y, cv=CV, scoring="accuracy")
    auc = cross_val_score(model, X, y, cv=CV, scoring="roc_auc")
    return float(acc.mean()), float(acc.std()), float(auc.mean())


def run(name, model, X, y, feature_set: str, tags: dict | None = None):
    with mlflow.start_run(run_name=f"{name}_{feature_set}"):
        mlflow.set_tag("model", name)
        mlflow.set_tag("feature_set", feature_set)
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        params = {k: str(v) for k, v in model.get_params().items()}
        mlflow.log_params(params)

        acc, std, auc = cv_metrics(model, X, y)
        mlflow.log_metric("cv_accuracy", acc)
        mlflow.log_metric("cv_std", std)
        mlflow.log_metric("cv_auc", auc)

        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model")

        print(f"  {name:20s} [{feature_set:12s}]  acc={acc:.4f} ±{std:.4f}  auc={auc:.4f}")
        return acc, model


def make_models():
    return [
        ("logreg", LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")),
        ("random_forest", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
        ("gradient_boost", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)),
        ("xgboost", XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                   use_label_encoder=False, eval_metric="logloss", random_state=42)),
        ("lightgbm", LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                     random_state=42, verbose=-1)),
    ]


def main():
    train = pd.read_csv("data/train.csv")
    y = train["Survived"].values

    mlflow.set_experiment(EXPERIMENT)

    print("=== Feature set: baseline ===")
    X_all, _, enc = build_features(train)
    X_base = X_all[BASELINE_COLS]
    results_base = {}
    for name, model in make_models():
        acc, fitted = run(name, model, X_base, y, "baseline")
        results_base[name] = (acc, fitted)

    print("\n=== Feature set: engineered ===")
    results_eng = {}
    for name, model in make_models():
        acc, fitted = run(name, model, X_all, y, "engineered")
        results_eng[name] = (acc, fitted)

    print("\n=== Feature set: reduced ===")
    X_red = X_all[REDUCED_COLS]
    for name, model in make_models():
        run(name, model, X_red, y, "reduced")

    print("\n=== Voting ensemble (engineered, top-3 by cv_accuracy) ===")
    top3 = sorted(results_eng.items(), key=lambda x: x[1][0], reverse=True)[:3]
    estimators = [(name, m) for name, (_, m) in top3]
    print(f"  constituents: {[n for n, _ in estimators]}")
    voting = VotingClassifier(estimators=estimators, voting="soft")
    run("voting", voting, X_all, y, "engineered")

    print("\nDone. Run `mlflow ui` to inspect results.")


if __name__ == "__main__":
    main()
