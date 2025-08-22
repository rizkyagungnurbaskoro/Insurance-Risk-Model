# Insurance Risk Model (XGBoost + KaggleHub)

This project builds a **binary classifier** to predict insurance risk using the **Car Insurance dataset** from Kaggle.  
The model is trained with **XGBoost**, and the dataset is fetched automatically via **KaggleHub** (no manual download required).

---

## 🚀 Quick Start

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model with XGBoost
make train_kaggle_xgb

# 4. Evaluate classification performance
make eval_kaggle

# 5. Generate visualizations (ROC, PR, calibration, confusion matrix, feature importance, lift/gain)
make viz_kaggle

# 6. Run predictions on the first 5 rows
python src/predict_xgb.py

## 📊 Results

Sample outputs after training and evaluation:

- **ROC Curve**  
<img width="905" height="721" alt="roc_curve" src="https://github.com/user-attachments/assets/af2ac4a1-dfd9-4eb2-b97e-155279943940" />

- **PR Curve**  
 <img width="905" height="721" alt="pr_curve" src="https://github.com/user-attachments/assets/9b4eaf93-65fc-4668-b196-d96eb56c6666" />

- **Calibration Curve**
<img width="905" height="721" alt="calibration_curve" src="https://github.com/user-attachments/assets/af637dbc-ff53-4372-abe7-33c4f8d09e9b" />
 
- **Confusion Matrix**  
<img width="877" height="721" alt="confusion_matrix_0 25" src="https://github.com/user-attachments/assets/665ee694-1ee4-4a11-a655-1f19a5d1002f" />

- **Feature Importance**  
<img width="1264" height="1259" alt="feature_importance" src="https://github.com/user-attachments/assets/d7dd6b2b-f685-402f-abc1-f1108e9e7384" />

- **Lift Curve**  
 <img width="905" height="721" alt="lift_curve" src="https://github.com/user-attachments/assets/2b284f0c-81a9-434f-bbff-1856eea9e74c" />

- **Gain Curve**  
 <img width="912" height="721" alt="gains_curve" src="https://github.com/user-attachments/assets/6a535e7f-ca3f-479e-9dc9-92120f1dec10" />

