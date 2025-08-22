# Insurance Risk Model (XGBoost + KaggleHub)

Binary classifier for insurance risk using the **Car Insurance** dataset from Kaggle, trained with **XGBoost**.
The dataset is fetched locally via **KaggleHub** (no manual download needed).

## Quick Start

```bash
# (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train with XGBoost
make train_kaggle_xgb

# Evaluate (classification report)
make eval_kaggle

# Visualize (ROC, PR, calibration, confusion matrix, feature importance, lift/gain)
make viz_kaggle

# Try predictions on first 5 rows
python src/predict_xgb.py

## Results

![ROC](docs/roc_curve.png)
![PR](docs/pr_curve.png)
![Calibration](docs/calibration_curve.png)
