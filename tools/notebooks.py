from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

def nb_new(cells, title="Python 3"):
    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    for cell in cells:
        if cell["type"] == "md":
            nb["cells"].append(nbf.v4.new_markdown_cell(cell["src"]))
        elif cell["type"] == "py":
            c = nbf.v4.new_code_cell(cell["src"])
            c["execution_count"] = 0
            c["outputs"] = []
            nb["cells"].append(c)
    nb["metadata"] = {
        "kernelspec": {"display_name": title, "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb

# 01 - Explore
cells_01 = [
    {"type": "md", "src": "# 01 · Explore Car Insurance Dataset\n\nQuick EDA on the Kaggle *Car Insurance* dataset (OUTCOME target)."},
    {"type": "py", "src": """import kagglehub, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

path = kagglehub.dataset_download("sagnik1511/car-insurance-data")
csvs = list(Path(path).rglob("*.csv"))
csvs
"""},
    {"type": "py", "src": """df = pd.read_csv(csvs[0])
df.head()"""},
    {"type": "md", "src": "## Basic info & summary"},
    {"type": "py", "src": """import pandas as pd
display(pd.DataFrame({"dtype": df.dtypes.astype(str), "nunique": df.nunique()}))
df.isna().mean().sort_values(ascending=False).head(20)"""},
    {"type": "md", "src": "## Target balance (`OUTCOME`)"},
    {"type": "py", "src": """assert "OUTCOME" in df.columns
ax = sns.countplot(x="OUTCOME", data=df)
ax.set_title("Target Distribution (OUTCOME)")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha='center', va='bottom')
plt.show()"""},
    {"type": "md", "src": "## Categorical distributions"},
    {"type": "py", "src": """cat_cols = [c for c in df.columns if df[c].dtype=='object' or df[c].nunique()<=12]
cat_cols = [c for c in cat_cols if c != "OUTCOME"]
n = min(6, len(cat_cols))
import math
rows = n
plt.figure(figsize=(14, 2.8*rows))
for i, col in enumerate(cat_cols[:n], 1):
    plt.subplot(rows, 1, i)
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
plt.show()"""},
    {"type": "md", "src": "## Numeric distributions & correlations"},
    {"type": "py", "src": """num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols].hist(figsize=(14, 10), bins=30)
plt.suptitle("Numeric Features Distribution"); plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap (numeric)"); plt.show()"""},
]

# 02 - Train XGBoost
cells_02 = [
    {"type": "md", "src": "# 02 · Train XGBoost Model\nTrain an XGBoost binary classifier on `OUTCOME`, evaluate metrics, and save artifacts."},
    {"type": "py", "src": """import kagglehub, pandas as pd, numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

def prepare_categoricals_for_xgb(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    for c in Xc.columns:
        if Xc[c].dtype == "object":
            Xc[c] = Xc[c].astype("category")
        if str(Xc[c].dtype).startswith("category"):
            Xc[c] = Xc[c].cat.add_categories(["__NA__"]).fillna("__NA__")
    return Xc

path = kagglehub.dataset_download("sagnik1511/car-insurance-data")
csv = list(Path(path).rglob("*.csv"))[0]
df = pd.read_csv(csv)

target = "OUTCOME"
y = (df[target].astype(str).str.lower()
        .map({"1":1,"0":0,"yes":1,"no":0,"true":1,"false":0})
        .fillna(df[target]).astype(int))
X = df.drop(columns=[target])
X = prepare_categoricals_for_xgb(X)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pos, neg = int(y_tr.sum()), len(y_tr)-int(y_tr.sum())
scale_pos_weight = float(neg/max(pos,1))

params = {
    "objective": "binary:logistic",
    "eval_metric": ["auc","logloss"],
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": 2.0,
    "alpha": 0.0,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42
}

dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
dte = xgb.DMatrix(X_te, label=y_te, enable_categorical=True)

booster = xgb.train(params, dtr, num_boost_round=1200,
                    evals=[(dtr,"train"), (dte,"valid")],
                    early_stopping_rounds=80, verbose_eval=False)

p = booster.predict(xgb.DMatrix(X_te, enable_categorical=True))
p_clip = np.clip(p, 1e-7, 1-1e-7)

metrics = {
    "roc_auc": float(roc_auc_score(y_te, p)),
    "pr_auc": float(average_precision_score(y_te, p)),
    "logloss": float(log_loss(y_te, p_clip)),
    "brier": float(brier_score_loss(y_te, p_clip))
}
metrics""" },
    {"type": "md", "src": "## Save model & metadata"},
    {"type": "py", "src": """out = Path("../models"); out.mkdir(exist_ok=True)
booster.save_model(str(out/"car_insurance_xgb.json"))

import json
meta = {
    "path": str(csv),
    "target": target,
    "features": list(X.columns),
    "class_balance": y.value_counts().to_dict(),
    "metrics": metrics
}
(out/"car_insurance_xgb_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("Saved model & meta to ../models/")"""},
]

# 03 - SHAP Interpretability (optional)
cells_03 = [
    {"type": "md", "src": "# 03 · Interpretability with SHAP (XGBoost)\nGlobal and local explanations using SHAP.\n\n**Install:** `pip install shap`"},
    {"type": "py", "src": """import kagglehub, pandas as pd, numpy as np
import xgboost as xgb, shap
from pathlib import Path
import matplotlib.pyplot as plt

def make_dm_ready(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    for c in Xc.columns:
        if Xc[c].dtype == "object":
            Xc[c] = Xc[c].astype("category")
        if str(Xc[c].dtype).startswith("category"):
            Xc[c] = Xc[c].cat.add_categories(["__NA__"]).fillna("__NA__")
    return Xc

path = kagglehub.dataset_download("sagnik1511/car-insurance-data")
csv = list(Path(path).rglob("*.csv"))[0]
df = pd.read_csv(csv)

target = "OUTCOME"
y = (df[target].astype(str).str.lower()
        .map({"1":1,"0":0,"yes":1,"no":0,"true":1,"false":0})
        .fillna(df[target]).astype(int))
X = df.drop(columns=[target])
Xc = make_dm_ready(X)

booster = xgb.Booster(); booster.load_model(str(Path("../models/car_insurance_xgb.json")))

explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(Xc)
shap_values[:1].shape  # sanity"""},
    {"type": "md", "src": "## SHAP summary plot"},
    {"type": "py", "src": """plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, Xc, show=False)
plt.tight_layout(); plt.show()"""},
    {"type": "md", "src": "## SHAP force plot (single row)"},
    {"type": "py", "src": """row_idx = 0
shap.force_plot(explainer.expected_value, shap_values[row_idx,:], Xc.iloc[row_idx,:], matplotlib=True)"""},
]

# write files
nb1 = nb_new(cells_01); (NB_DIR / "01_explore_dataset.ipynb").write_text(nbf.writes(nb1), encoding="utf-8")
nb2 = nb_new(cells_02); (NB_DIR / "02_train_model_xgb.ipynb").write_text(nbf.writes(nb2), encoding="utf-8")
nb3 = nb_new(cells_03); (NB_DIR / "03_interpretability_shap.ipynb").write_text(nbf.writes(nb3), encoding="utf-8")

print("Created notebooks in:", NB_DIR)
