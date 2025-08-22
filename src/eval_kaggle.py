from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from load_kaggle_car_insurance import load_kaggle_car_insurance

MODEL_FILE = Path("models/car_insurance_xgb.json")

def make_dm_ready(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    for c in Xc.columns:
        if pd.api.types.is_object_dtype(Xc[c]):
            Xc[c] = Xc[c].astype("category")
        if pd.api.types.is_categorical_dtype(Xc[c]):
            Xc[c] = Xc[c].cat.add_categories(["__NA__"]).fillna("__NA__")
        # numeric dtypes are fine as-is
    return Xc

def main(threshold: float = 0.25):
    # hard-code correct target for this dataset
    X, y, meta = load_kaggle_car_insurance(target_override="OUTCOME")

    # ensure valid dtypes for XGBoost
    Xc = make_dm_ready(X)

    booster = xgb.Booster()
    booster.load_model(str(MODEL_FILE))

    dmat = xgb.DMatrix(Xc, enable_categorical=True)
    proba = booster.predict(dmat)
    pred = (proba >= threshold).astype(int)

    print("Target:", meta["target"])
    print("Class balance:", meta["class_balance"])
    print(f"\nClassification report @ threshold={threshold:.2f}\n")
    print(classification_report(y, pred, digits=4))

if __name__ == "__main__":
    main()
