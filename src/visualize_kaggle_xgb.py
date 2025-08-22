from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

from load_kaggle_car_insurance import load_kaggle_car_insurance

OUT = Path("models"); OUT.mkdir(exist_ok=True)
MODEL = OUT / "car_insurance_xgb.json"

def savefig(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()

def make_dm_ready(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    for c in Xc.columns:
        if pd.api.types.is_object_dtype(Xc[c]):
            Xc[c] = Xc[c].astype("category")
        if pd.api.types.is_categorical_dtype(Xc[c]):
            Xc[c] = Xc[c].cat.add_categories(["__NA__"]).fillna("__NA__")
    return Xc

def main():
    # use correct target
    X, y, meta = load_kaggle_car_insurance(target_override="OUTCOME")
    Xc = make_dm_ready(X)

    booster = xgb.Booster(); booster.load_model(str(MODEL))
    dmat = xgb.DMatrix(Xc, enable_categorical=True)
    proba = booster.predict(dmat)
    proba_clip = np.clip(proba, 1e-7, 1 - 1e-7)

    # ROC
    fpr, tpr, _ = roc_curve(y, proba); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); savefig(OUT/"roc_curve.png")

    # PR
    prec, rec, _ = precision_recall_curve(y, proba); ap = average_precision_score(y, proba)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve"); plt.legend(); savefig(OUT/"pr_curve.png")

    # Calibration
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    plt.figure(); plt.plot(mean_pred, frac_pos, marker="o"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve"); savefig(OUT/"calibration_curve.png")

    # Confusion @ 0.25
    thr = 0.25; pred = (proba >= thr).astype(int); cm = confusion_matrix(y, pred)
    plt.figure(); plt.imshow(cm); plt.title(f"Confusion Matrix @ {thr:.2f}"); plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); savefig(OUT/f"confusion_matrix_{thr:.2f}.png")

    # Feature importance (gain)
    imp = booster.get_score(importance_type="gain")
    if imp:
        fi = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())}).sort_values("importance")
        plt.figure(figsize=(8, 8)); plt.barh(fi["feature"], fi["importance"])
        plt.title("Feature Importance (gain)"); plt.tight_layout(); savefig(OUT/"feature_importance.png")

    # Gains / Lift
    order = np.argsort(-proba); y_sorted = y.values[order]
    n = len(y_sorted); pos_total = y.sum()
    bins = 10; cuts = np.linspace(0, n, bins+1, dtype=int)
    cum_pos = np.cumsum([y_sorted[cuts[i-1]:cuts[i]].sum() for i in range(1, len(cuts))])
    frac = cuts[1:]/n; gain = cum_pos/pos_total; lift = gain/np.maximum(frac, 1e-9)
    plt.figure(); plt.plot(frac*100, gain*100); plt.xlabel("% population"); plt.ylabel("% positives captured")
    plt.title("Cumulative Gains"); savefig(OUT/"gains_curve.png")
    plt.figure(); plt.plot(frac*100, lift); plt.xlabel("% population"); plt.ylabel("Lift")
    plt.title("Lift Chart"); savefig(OUT/"lift_curve.png")

    # Metrics JSON
    from sklearn.metrics import accuracy_score
    metrics = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "logloss": float(log_loss(y, proba_clip)),
        "brier": float(brier_score_loss(y, proba_clip)),
        "avg_precision": float(ap),
        "threshold": thr,
        "accuracy_at_threshold": float(accuracy_score(y, pred))
    }
    (OUT/"viz_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved figures and metrics to:", OUT.resolve())

if __name__ == "__main__":
    main()
