import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Mme": "Mrs",
    "Capt": "Rare", "Sir": "Rare",
}

CATEGORICAL_COLS = ["Sex", "Embarked", "Title", "AgeGroup", "FareBand", "Deck"]


def _extract_title(name: str) -> str:
    match = re.search(r",\s*([^.]+)\.", name)
    if match:
        raw = match.group(1).strip()
        return TITLE_MAP.get(raw, "Rare")
    return "Rare"


def build_features(df: pd.DataFrame, encoder: OrdinalEncoder | None = None, fit: bool = True):
    """
    Engineer features from raw Titanic DataFrame.
    Returns (transformed_df, feature_columns, fitted_encoder).
    Pass fit=False and a fitted encoder when transforming test data.
    """
    df = df.copy()

    # ── imputation ──
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # ── engineered features ──
    df["Title"] = df["Name"].apply(_extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "U")

    ticket_freq = df["Ticket"].value_counts()
    df["TicketFreq"] = df["Ticket"].map(ticket_freq)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Adult", "Middle", "Senior"],
    ).astype(str)

    # FareBand: use fixed quantiles computed on training distribution
    if fit:
        df["FareBand"], fare_bins = pd.qcut(df["Fare"], 4, retbins=True, labels=["Q1", "Q2", "Q3", "Q4"])
        df.attrs["fare_bins"] = fare_bins.tolist()
    else:
        fare_bins = encoder.fare_bins_ if hasattr(encoder, "fare_bins_") else [0, 7.91, 14.454, 31.0, 600]
        df["FareBand"] = pd.cut(
            df["Fare"],
            bins=fare_bins,
            labels=["Q1", "Q2", "Q3", "Q4"],
            include_lowest=True,
        ).astype(str)
    df["FareBand"] = df["FareBand"].astype(str)

    # ── encode categoricals ──
    if encoder is None:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    if fit:
        df[CATEGORICAL_COLS] = encoder.fit_transform(df[CATEGORICAL_COLS])
        encoder.fare_bins_ = df.attrs.get("fare_bins", [0, 7.91, 14.454, 31.0, 600])
    else:
        df[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS])

    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
        "Title", "FamilySize", "IsAlone", "HasCabin", "Deck",
        "TicketFreq", "AgeGroup", "FareBand",
    ]

    return df[feature_cols], feature_cols, encoder


BASELINE_COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
REDUCED_COLS = [
    "Pclass", "Sex", "Age", "Fare", "Embarked",
    "Title", "FamilySize", "IsAlone", "HasCabin", "TicketFreq",
]


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    X, cols, enc = build_features(train)
    print(f"train shape: {X.shape}")
    print(f"features: {cols}")
    print(X.head())

    test = pd.read_csv("data/test.csv")
    X_test, _, _ = build_features(test, encoder=enc, fit=False)
    print(f"\ntest shape: {X_test.shape}")
