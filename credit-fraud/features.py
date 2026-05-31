import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ── Column configuration (fill in after EDA) ──────────────────────────────────
# Run `python features.py` after downloading data to inspect raw column names.
CATEGORICAL_COLS: list[str] = []   # columns to ordinal-encode
NUMERIC_COLS: list[str] = []       # numeric feature columns (used to fit scaler)
BASELINE_COLS: list[str] = []      # raw numeric + encoded cats, no engineering
ENGINEERED_COLS: list[str] = []    # all features including transforms / interactions
# ──────────────────────────────────────────────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    encoder: OrdinalEncoder | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = True,
):
    """
    Engineer features from raw credit-fraud DataFrame.
    Returns (X_df, feature_cols, fitted_encoder, fitted_scaler).
    Pass fit=False with a fitted encoder+scaler when transforming test data.
    Caller is responsible for dropping the target and ID columns before calling.
    """
    df = df.copy()

    # ── 1. Imputation ──────────────────────────────────────────────────────────
    # TODO: replace with column-specific strategy after EDA reveals missingness.
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # ── 2. Log-transform skewed numerics ───────────────────────────────────────
    # TODO: identify high-skew columns from EDA and apply np.log1p.
    # Example: df["amount_log"] = np.log1p(df["amount"])

    # ── 3. Interaction / ratio features ────────────────────────────────────────
    # TODO: add domain-driven interactions after EDA.
    # Example: df["debt_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)

    # ── 4. Bin high-cardinality numerics ──────────────────────────────────────
    # TODO: identify binning candidates from EDA; store cut bins on encoder
    # so the same bins are applied to test data. Mirror the Titanic fare_bins_ pattern:
    #   if fit:
    #       df["amount_band"], bins = pd.qcut(df["amount"], 4, retbins=True,
    #                                         labels=["Q1","Q2","Q3","Q4"])
    #       encoder.amount_bins_ = bins.tolist()
    #   else:
    #       df["amount_band"] = pd.cut(df["amount"], bins=encoder.amount_bins_,
    #                                  labels=["Q1","Q2","Q3","Q4"],
    #                                  include_lowest=True).astype(str)

    # ── 5. Encode categoricals ─────────────────────────────────────────────────
    if encoder is None:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if CATEGORICAL_COLS:
        if fit:
            df[CATEGORICAL_COLS] = encoder.fit_transform(df[CATEGORICAL_COLS].astype(str))
        else:
            df[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS].astype(str))
    elif fit:
        encoder.fit([[]])  # no-op fit so encoder can be pickled / reused

    # ── 6. StandardScaler (returned for LR pipelines in train.py) ─────────────
    if scaler is None:
        scaler = StandardScaler()
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    if numeric_present and fit:
        scaler.fit(df[numeric_present])

    feature_cols = (
        ENGINEERED_COLS
        if ENGINEERED_COLS
        else df.select_dtypes(include="number").columns.tolist()
    )
    return df[feature_cols], feature_cols, encoder, scaler


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    train = pd.read_csv("data/train.csv")
    print(f"train shape: {train.shape}")
    print(f"\nColumns:\n{list(train.columns)}")
    print(f"\nDtypes:\n{train.dtypes}")
    missing = train.isnull().sum()
    print(f"\nMissing values:\n{missing[missing > 0]}")
    print(f"\nSample:\n{train.head(3).to_string()}")

    if ENGINEERED_COLS:
        X, cols, enc, scl = build_features(train)
        print(f"\nFeature matrix: {X.shape}")
        print(f"Features: {cols}")
        print(f"Nulls: {X.isnull().sum().sum()}")

        test = pd.read_csv("data/test.csv")
        X_test, _, _, _ = build_features(test, encoder=enc, scaler=scl, fit=False)
        print(f"Test feature matrix: {X_test.shape}")
    else:
        print("\nNOTE: ENGINEERED_COLS is empty — fill in after EDA, then re-run.")
