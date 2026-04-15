"""
City Pulse — Data Loading & Preprocessing
==========================================
Handles loading, cleaning, validation, and feature engineering
for the urban liveability dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional

# Canonical paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_CSV = DATA_DIR / "cities.csv"

# Feature groups
NUMERIC_FEATURES = [
    "air_quality",
    "green_space",
    "transit_score",
    "safety_index",
    "cost_of_living",
    "noise_level",
]
TARGET = "liveability_score"
CATEGORICAL_FEATURES = ["country", "continent"]
FEATURE_DISPLAY_NAMES = {
    "air_quality": "Air Quality",
    "green_space": "Green Space",
    "transit_score": "Transit Score",
    "safety_index": "Safety Index",
    "cost_of_living": "Cost of Living",
    "noise_level": "Noise Level",
}


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw CSV and return as a DataFrame."""
    path = path or RAW_CSV
    df = pd.read_csv(path)
    print(f"[loader] Loaded {len(df)} rows × {df.shape[1]} columns from '{path.name}'")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run basic data-quality checks. Raises ValueError on fatal issues,
    prints warnings for minor ones.

    Checks:
    - Required columns present
    - No duplicate city names
    - All numeric features in [0, 100]
    - No nulls in key columns
    """
    required = ["city"] + NUMERIC_FEATURES + [TARGET]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    dupes = df[df.duplicated("city", keep=False)]["city"].tolist()
    if dupes:
        print(f"[validate] WARNING — duplicate city names: {dupes}")

    nulls = df[required].isnull().sum()
    if nulls.any():
        print(f"[validate] WARNING — null values:\n{nulls[nulls > 0]}")

    for col in NUMERIC_FEATURES + [TARGET]:
        out_of_range = df[(df[col] < 0) | (df[col] > 100)]["city"].tolist()
        if out_of_range:
            print(f"[validate] WARNING — '{col}' out of [0,100] for: {out_of_range}")

    print(f"[validate] Passed basic checks on {len(df)} rows.")
    return df


# ─── Cleaning ─────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning steps:
    - Strip whitespace from string columns
    - Clip numeric features to [0, 100]
    - Drop fully-null rows
    - Reset index
    """
    df = df.copy()

    # Strip strings
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # Clip numerics
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = df[col].clip(0, 100)

    df = df.dropna(subset=["city"] + NUMERIC_FEATURES).reset_index(drop=True)
    print(f"[clean] Dataset shape after cleaning: {df.shape}")
    return df


# ─── Feature Engineering ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional features for richer analysis.

    New columns:
    - env_score    : mean of air_quality, green_space (environment proxy)
    - social_score : mean of transit_score, safety_index (social infrastructure)
    - affordability: inverted cost_of_living (higher = more affordable)
    - quiet_score  : inverted noise_level
    - composite    : unweighted mean of all 6 raw features
    """
    df = df.copy()
    df["env_score"] = df[["air_quality", "green_space"]].mean(axis=1).round(1)
    df["social_score"] = df[["transit_score", "safety_index"]].mean(axis=1).round(1)
    df["affordability"] = (100 - df["cost_of_living"]).round(1)
    df["quiet_score"] = (100 - df["noise_level"]).round(1)
    df["composite"] = df[NUMERIC_FEATURES].mean(axis=1).round(1)
    return df


# ─── Scaling ──────────────────────────────────────────────────────────────────

def scale_features(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = "standard",
) -> Tuple[pd.DataFrame, object]:
    """
    Scale numeric features.

    Parameters
    ----------
    df       : cleaned DataFrame
    features : list of column names to scale (defaults to NUMERIC_FEATURES)
    method   : 'standard' (z-score) or 'minmax'

    Returns
    -------
    df_scaled : DataFrame with scaled columns (originals replaced)
    scaler    : fitted scaler object (for inverse_transform later)
    """
    features = features or NUMERIC_FEATURES
    df_scaled = df.copy()

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df_scaled[features] = scaler.fit_transform(df[features])

    print(f"[scale] Applied '{method}' scaling to {len(features)} features.")
    return df_scaled, scaler


# ─── Train / Test Split ───────────────────────────────────────────────────────

def split(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into train/test sets.

    Returns X_train, X_test, y_train, y_test.
    """
    features = features or NUMERIC_FEATURES
    X = df[features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(
        f"[split] Train: {len(X_train)} rows | Test: {len(X_test)} rows "
        f"({test_size*100:.0f}% test)"
    )
    return X_train, X_test, y_train, y_test


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def load_pipeline(
    path: Optional[Path] = None,
    scale: bool = True,
    engineer_features: bool = True,
) -> dict:
    """
    One-call convenience function that runs the full loading pipeline.

    Returns a dict with keys:
        raw, clean, df, X_train, X_test, y_train, y_test, scaler
    """
    raw = load_raw(path)
    validate(raw)
    cleaned = clean(raw)
    if engineer_features:
        cleaned = add_features(cleaned)

    result = {"raw": raw, "clean": cleaned, "df": cleaned}

    if scale:
        df_scaled, scaler = scale_features(cleaned)
        result["df_scaled"] = df_scaled
        result["scaler"] = scaler
        X_tr, X_te, y_tr, y_te = split(df_scaled)
    else:
        X_tr, X_te, y_tr, y_te = split(cleaned)
        result["scaler"] = None

    result.update(
        {"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te}
    )
    return result


# ─── Quick summary helper ─────────────────────────────────────────────────────

def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a descriptive statistics table for numeric features."""
    return df[NUMERIC_FEATURES + [TARGET]].describe().round(2)
