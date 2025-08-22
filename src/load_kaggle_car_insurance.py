"""
KaggleHub loader for the 'sagnik1511/car-insurance-data' dataset.

- Finds a CSV inside the KaggleHub folder.
- Resolves the binary target column automatically or via TARGET_NAME env var.
- Marks categorical columns (objects + low-cardinality ints).
- Returns X (features), y (0/1), and meta (path, target, categorical columns, class balance).
"""

from __future__ import annotations
from pathlib import Path
import os
import re
from typing import Tuple, Dict, List

import pandas as pd
import numpy as np
import kagglehub

# Common candidates across car insurance datasets
CANDIDATE_TARGETS = [
    "CarInsurance", "car_insurance", "Car_Insurance",
    "Response", "response", "Purchase", "purchase",
    "y", "target", "bought_insurance",
    "Claim", "claim", "Claim_Flag", "CLAIM_FLAG", "fraud_reported"
]

def _find_csv(root: Path) -> Path:
    csvs = list(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {root}")
    # Prefer obvious names if present
    prefer = {"car_insurance.csv", "Car_Insurance.csv", "car insurance claim.csv"}
    for p in csvs:
        if p.name in prefer:
            return p
    return csvs[0]

def _normalize_target(y: pd.Series) -> pd.Series:
    if y.dtype == "O":
        s = y.astype(str).str.strip().str.lower()
        if set(s.unique()) <= {"yes", "no"}:
            return (s == "yes").astype(int)
        if set(s.unique()) <= {"y", "n"}:
            return (s == "y").astype(int)
        if set(s.unique()) <= {"true", "false"}:
            return (s == "true").astype(int)
    y = pd.to_numeric(y, errors="coerce")
    if not set(pd.Series(y.dropna().unique())) <= {0, 1}:
        y = (y > 0).astype(int)
    return y.fillna(0).astype(int)

def _infer_target_name(df: pd.DataFrame) -> str | None:
    lowers = [c.lower() for c in df.columns]
    for cand in CANDIDATE_TARGETS:
        if cand.lower() in lowers:
            return df.columns[lowers.index(cand.lower())]
    # Fallback: any exactly-binary column
    for c in df.columns:
        vals = pd.Series(df[c]).dropna().unique()
        if len(vals) == 2:
            return c
    return None

def _drop_identifier_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop_like = [c for c in df.columns if re.search(r"(^id$)|customer|client", c, re.I)]
    return df.drop(columns=list(set(drop_like)), errors="ignore")

def _mark_categoricals(X: pd.DataFrame) -> List[str]:
    cats: List[str] = [c for c in X.columns if X[c].dtype == "O"]
    for c in X.columns:
        if X[c].dtype != "O":
            nun = X[c].nunique(dropna=True)
            if 2 <= nun <= 12:
                X[c] = X[c].astype("category")
                cats.append(c)
    # de-dup keep order
    seen, out = set(), []
    for c in cats:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def load_kaggle_car_insurance(target_override: str | None = None):
    """
    Returns:
        X: DataFrame
        y: Series (0/1 int)
        meta: dict { path, target, categorical_cols, class_balance }
    You can set an explicit target via `target_override` or env var `TARGET_NAME`.
    """
    root = Path(kagglehub.dataset_download("sagnik1511/car-insurance-data"))
    csv_path = _find_csv(root)
    df = pd.read_csv(csv_path)

    target = target_override or os.getenv("TARGET_NAME") or _infer_target_name(df)
    if target is None or target not in df.columns:
        raise ValueError(
            f"Could not resolve target column. "
            f"Set TARGET_NAME env var or pass target_override. "
            f"Available columns: {list(df.columns)}"
        )

    y = _normalize_target(df[target])
    if y.nunique(dropna=True) < 2:
        raise ValueError(
            f"Target '{target}' has one class only: {y.value_counts().to_dict()}."
        )

    df = _drop_identifier_cols(df)
    X = df.drop(columns=[target])

    categorical_cols = _mark_categoricals(X)

    meta = {
        "path": str(csv_path),
        "target": target,
        "categorical_cols": categorical_cols,
        "class_balance": y.value_counts().to_dict()
    }
    return X, y.astype(int), meta

# Convenience helper if you want to inspect columns quickly
def peek_columns():
    root = Path(kagglehub.dataset_download("sagnik1511/car-insurance-data"))
    csv_path = _find_csv(root)
    df = pd.read_csv(csv_path)
    print("CSV:", csv_path)
    print("Columns:", list(df.columns))
    print(df.head())
