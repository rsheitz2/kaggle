import sys
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path

from features import build_features, BASELINE_COLS, ENGINEERED_COLS

TARGET_COL = "target"    # TODO: update to actual target column name
ID_COL = "id"            # TODO: update to actual ID column name
ID_COLS: list[str] = []  # TODO: update other columns to drop (same as train.py)


def main():
    mlflow.set_experiment("credit-fraud")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("credit-fraud")
    if experiment is None:
        sys.exit("No 'credit-fraud' experiment found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.cv_pr_auc DESC"],
        max_results=1,
    )
    if not runs:
        sys.exit("No runs found. Run train.py first.")

    best = runs[0]
    run_id = best.info.run_id
    model_tag = best.data.tags.get("model", "unknown")
    feat_set = best.data.tags.get("feature_set", "all")
    threshold = float(best.data.params.get("optimal_threshold", 0.5))
    pr_auc = best.data.metrics.get("cv_pr_auc", 0)
    f1_tuned = best.data.metrics.get("cv_f1_tuned", 0)
    print(
        f"Best run: {run_id[:8]}"
        f"  model={model_tag}"
        f"  feat={feat_set}"
        f"  cv_pr_auc={pr_auc:.4f}"
        f"  cv_f1_tuned={f1_tuned:.4f}"
        f"  threshold={threshold:.2f}"
    )

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    train_raw = pd.read_csv("data/train.csv")
    test_raw = pd.read_csv("data/test.csv")
    test_ids = test_raw[ID_COL]

    drop_cols = [TARGET_COL] + ID_COLS
    _, _, enc, scl = build_features(train_raw.drop(columns=drop_cols, errors="ignore"))

    test_drop = [c for c in ID_COLS if c in test_raw.columns]
    X_test, _, _, _ = build_features(
        test_raw.drop(columns=test_drop, errors="ignore"),
        encoder=enc, scaler=scl, fit=False,
    )

    if feat_set == "baseline" and BASELINE_COLS:
        X_test = X_test[BASELINE_COLS]
    elif feat_set == "engineered" and ENGINEERED_COLS:
        X_test = X_test[ENGINEERED_COLS]

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    print(f"Predicted positive rate: {preds.mean():.2%}")

    out_dir = Path("submissions")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"submission_{run_id[:8]}.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds}).to_csv(out_path, index=False)
    print(f"Submission written: {out_path}  ({len(preds)} rows)")


if __name__ == "__main__":
    main()
