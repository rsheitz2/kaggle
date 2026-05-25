import sys
import pandas as pd
import mlflow
from pathlib import Path

from features import build_features


def main():
    mlflow.set_experiment("titanic")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("titanic")
    if experiment is None:
        sys.exit("No 'titanic' experiment found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.cv_accuracy DESC"],
        max_results=1,
    )
    if not runs:
        sys.exit("No runs found. Run train.py first.")

    best = runs[0]
    run_id = best.info.run_id
    acc = best.data.metrics.get("cv_accuracy", 0)
    model_name = best.data.tags.get("model", "unknown")
    feat_set = best.data.tags.get("feature_set", "unknown")
    print(f"Best run: {run_id[:8]}  model={model_name}  feature_set={feat_set}  cv_acc={acc:.4f}")

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    test_raw = pd.read_csv("data/test.csv")
    passenger_ids = test_raw["PassengerId"]

    # Re-build encoder from training data to get correct fit
    train_raw = pd.read_csv("data/train.csv")
    _, _, enc = build_features(train_raw)
    X_test, feat_cols, _ = build_features(test_raw, encoder=enc, fit=False)

    # Match feature set used by the best run
    if feat_set == "baseline":
        from features import BASELINE_COLS
        X_test = X_test[BASELINE_COLS]
    elif feat_set == "reduced":
        from features import REDUCED_COLS
        X_test = X_test[REDUCED_COLS]

    preds = model.predict(X_test)

    out_dir = Path("submissions")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"submission_{run_id[:8]}.csv"
    pd.DataFrame({"PassengerId": passenger_ids, "Survived": preds}).to_csv(out_path, index=False)
    print(f"Submission written: {out_path}  ({len(preds)} rows)")


if __name__ == "__main__":
    main()
