from pathlib import Path
import pandas as pd
import xgboost as xgb
from load_kaggle_car_insurance import load_kaggle_car_insurance

MODEL_FILE = Path("models/car_insurance_xgb.json")

def main(n: int = 5, threshold: float = 0.25):
    X, y, meta = load_kaggle_car_insurance()
    booster = xgb.Booster(); booster.load_model(str(MODEL_FILE))

    d = xgb.DMatrix(X.iloc[:n], enable_categorical=True)
    proba = booster.predict(d)
    pred = (proba >= threshold).astype(int)

    out = X.iloc[:n].copy()
    out["probability"] = proba
    out["predicted"] = pred
    out["actual"] = y.iloc[:n].values
    print(out)

if __name__ == "__main__":
    main()
