import sys, os
sys.path.insert(0, r"c:\Users\thanu\Minor degree project")
sys.stdout.reconfigure(encoding='utf-8')

import pickle, shap, numpy as np

MODEL_DIR = r"c:\Users\thanu\Minor degree project\backend\ml"
MODEL_PATH = os.path.join(MODEL_DIR, "phishguard_model.pkl")
EXPLAINER_PATH = os.path.join(MODEL_DIR, "shap_explainer.pkl")

print("Loading XGBoost model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Building SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)

print("Saving SHAP explainer...")
with open(EXPLAINER_PATH, "wb") as f:
    pickle.dump(explainer, f)

print(f"SHAP explainer saved -> {EXPLAINER_PATH}")
print(f"File size: {os.path.getsize(EXPLAINER_PATH):,} bytes")
print("Done!")
