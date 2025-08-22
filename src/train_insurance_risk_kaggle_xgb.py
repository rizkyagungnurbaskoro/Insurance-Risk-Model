from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
import xgboost as xgb

from load_kaggle_car_insurance import load_kaggle_car_insurance

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "car_insurance_xgb.json"
META_FILE  = MODEL_DIR / "car_insurance_xgb_meta.json"

def prepare_categoricals_for_xgb(X: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    Xc = X.copy()
    for c in categorical_cols:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype("category")
            Xc[c] = Xc[c].cat.add_categories(["__NA__"]).fillna("__NA__")
    return Xc

def main():
    # Explicitly use OUTCOME as the target for this dataset
    X, y, meta = load_kaggle_car_insurance(target_override="OUTCOME")

    X = prepare_categoricals_for_xgb(X, meta["categorical_cols"])
    y = y.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = int(y_tr.sum()); neg = len(y_tr) - pos
    scale_pos_weight = float(neg / max(pos, 1))

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 2.0,
        "alpha": 0.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        # NOTE: do NOT put 'enable_categorical' here; pass it to DMatrix instead.
    }

    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_te, label=y_te, enable_categorical=True)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=1200,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=80,
        verbose_eval=False,
    )

    p = booster.predict(xgb.DMatrix(X_te, enable_categorical=True))
    # Clip for numerical stability (since sklearn.log_loss no longer has 'eps')
    p_clip = np.clip(p, 1e-7, 1 - 1e-7)

    metrics = {
        "roc_auc": float(roc_auc_score(y_te, p)),
        "pr_auc": float(average_precision_score(y_te, p)),
        "logloss": float(log_loss(y_te, p_clip)),
        "brier": float(brier_score_loss(y_te, p_clip)),
    }
    print(json.dumps({
        "path": meta["path"],
        "target": meta["target"],
        "class_balance": meta["class_balance"],
        "metrics": metrics
    }, indent=2))

    booster.save_model(str(MODEL_FILE))
    META_FILE.write_text(json.dumps({
        **meta,
        "features": list(X.columns),
        "metrics": metrics
    }, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_FILE}")

if __name__ == "__main__":
    main()
