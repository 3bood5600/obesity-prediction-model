# Obesity Prediction Model

A complete, notebook-driven machine learning pipeline to predict obesity categories (7 classes) from lifestyle and biometric data. The project covers data loading, EDA, preprocessing, feature engineering, model training, hyperparameter tuning, and artifact packaging for reuse.

- Best tuned model: XGBoost
- Best test accuracy: ~0.903 (see `models/tuned/model_info.json`)
- Frameworks: scikit-learn, XGBoost, imbalanced-learn, pandas, NumPy, matplotlib, seaborn

---

## Repository layout

- `data/` — input and intermediate datasets
  - `Obesity_data.csv` (raw), `loaded_data.csv`, `preprocessed_data.csv`, `engineered_features.csv`, `selected_features.csv`, `target_classes.json`
- `notebooks/` — end-to-end pipeline in notebooks (run in numeric order)
  - `00_Data_Loading.ipynb` → `05_Hyperparameter_Tuning_Evaluation.ipynb`
- `models/` — saved models and metadata
  - Base models: `*_model.pkl`, scaler: `feature_scaler.pkl`, feature names/info
  - Tuned: `tuned/*.pkl`, `tuned/model_info.json`, `tuned/tuned_results.csv`, `tuned/baseline_vs_tuned_comparison.csv`, `tuned/best_hyperparameters.txt`
- `results/` — EDA visuals, preprocessing details, feature selection, model comparisons
- `UI/` — placeholder for an optional app surface (Streamlit listed in deps)
- `requirements.txt` — project dependencies
- `LICENSE.txt`

---

## Quick start (Windows PowerShell)

Prerequisites:
- Python 3.9+ recommended
- Git, build tools suitable for your Python/XGBoost environment

Setup:

```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Data:
- Ensure `data/Obesity_data.csv` exists (already included).

Run the pipeline:
1. Open the `notebooks/` folder in your IDE.
2. Execute notebooks in order:
   - 00_Data_Loading.ipynb
   - 01_EDA.ipynb
   - 02_Data_Preprocessing.ipynb
   - 03_Feature_Engineering.ipynb
   - 04_Model_Training.ipynb
   - 05_Hyperparameter_Tuning_Evaluation.ipynb
3. Artifacts will be written under `models/` and `results/`.

---

## Reproducing training and tuning

- Baseline training creates base models in `models/` and baseline metrics in `results/model_training/`.
- Hyperparameter tuning evaluates Random Forest, Logistic Regression, Polynomial Regression (degree 3 via pipeline), and XGBoost using stratified 5-fold CV and weighted F1 scoring. Outputs:
  - Best estimators saved to `models/tuned/*.pkl`
  - Summary CSVs and best hyperparameters in `models/tuned/`
  - `models/tuned/model_info.json` indicates the best tuned model and metrics

Key preprocessing details live in `results/preprocessing_info.json` and `models/feature_info.json` (feature names, target classes, scaling/encoding).

---

## Using the trained model in your code

Load the best tuned model and required metadata, then prepare inputs with the same columns and preprocessing used in training.

```python
import json
import joblib
import pandas as pd
from pathlib import Path

repo = Path('.')
info = json.loads((repo/'models/tuned/model_info.json').read_text())
best_name = info['best_model']
model_path = Path(info['model_paths'][best_name]).resolve()
model = joblib.load(model_path)

# Optional scaler (for linear models); tree/XGBoost typically use raw engineered features
scaler = joblib.load(repo/'models/feature_scaler.pkl')
feature_info = json.loads((repo/'models/feature_info.json').read_text())
feature_names = feature_info['feature_names']

# Example: create a DataFrame with exactly the same columns
X = pd.read_csv(repo/'data/selected_features.csv').drop(columns=['NObeyesdad_encoded'], errors='ignore')
X = X[feature_names]  # ensure correct column order

# If your model requires scaling, apply it; XGBoost/RandomForest often do not
try:
    X_pred = scaler.transform(X)
except Exception:
    X_pred = X

y_prob = model.predict_proba(X_pred)
y_pred = model.predict(X_pred)
print(y_pred[:5])
```

Map predicted class indices to human-readable classes using `data/target_classes.json`.

---

## Results snapshot

- Best model (tuned): XGBoost
- Best test accuracy: ~0.903 (see `models/tuned/model_info.json`)
- Comparison of baseline vs tuned is available in `models/tuned/baseline_vs_tuned_comparison.csv`.

---

## Notes and troubleshooting

- Keep your input feature columns identical to those used in training (`models/feature_info.json`).
- For new data, replicate preprocessing (encoding and scaling) consistent with `results/preprocessing_info.json`.
- XGBoost may require a compatible compiler/runtime in some environments; update to the latest `xgboost` if you encounter build errors.

---

## License

This project is licensed under the terms in `LICENSE.txt`.

---

## Acknowledgments

- scikit-learn, XGBoost, imbalanced-learn, pandas, NumPy, matplotlib, seaborn.
