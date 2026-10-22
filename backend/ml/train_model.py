"""
PhishGuard Pro — Model Training Script
=======================================
Trains the hybrid XGBoost model on combined URL + content features.
Generates and saves a SHAP TreeExplainer for real-time explanations.

Usage:
    python backend/ml/train_model.py

Dataset options (in order of preference):
  1. datasets/malicious_urls.csv  — place your downloaded dataset here
  2. Auto-download from GitHub (rlilojr/Detecting-Malicious-URL)
  3. Synthetic demo dataset (for quick testing without real data)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.app.models.url_features import extract_url_features, FEATURE_NAMES
from backend.app.models.char_cnn import get_typosquat_score

MODEL_DIR    = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(MODEL_DIR, "phishguard_model.pkl")
EXPLAINER_PATH = os.path.join(MODEL_DIR, "shap_explainer.pkl")
DATASET_PATH = os.path.join(os.path.dirname(MODEL_DIR), "datasets", "malicious_urls.csv")

CONTENT_FEATURE_DEFAULTS = {
    "content_fetch_success": 0, "has_login_form": 0, "password_field_count": 0,
    "form_count": 0, "iframe_count": 0, "hidden_element_count": 0,
    "external_link_ratio": 0.0, "favicon_domain_mismatch": 0,
    "has_right_click_disable": 0, "has_popup_script": 0,
    "external_script_ratio": 0.0, "page_title_brand_mismatch": 0,
    "meta_refresh_redirect": 0, "copyright_year_missing": 0,
    "suspicious_action_url": 0,
}

ALL_FEATURE_NAMES = FEATURE_NAMES + list(CONTENT_FEATURE_DEFAULTS.keys()) + ["typosquat_score"]


def generate_synthetic_dataset(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate a synthetic labeled dataset for demo/testing purposes.
    Uses heuristic rules to assign labels — replace with real data for production.
    """
    print(f"[INFO] Generating {n_samples} synthetic samples...")

    legit_tlds = [".com", ".org", ".net", ".edu", ".gov"]
    phish_tlds = [".tk", ".ml", ".ga", ".cf", ".xyz"]

    brands    = ["google", "paypal", "amazon", "apple", "microsoft", "facebook"]
    typosquats = ["g00gle", "paypa1", "amaz0n", "app1e", "micros0ft", "faceb00k"]

    rows, labels = [], []

    # Legitimate URLs
    for _ in range(n_samples // 2):
        brand  = np.random.choice(brands)
        tld    = np.random.choice(legit_tlds)
        path   = "/".join(np.random.choice(["about", "contact", "home", "products", "services"],
                                            size=np.random.randint(1, 3)))
        url = f"https://www.{brand}{tld}/{path}"
        rows.append(url)
        labels.append(0)

    # Phishing URLs
    for _ in range(n_samples // 2):
        style = np.random.randint(0, 5)
        if style == 0:
            # IP address
            ip = ".".join(str(np.random.randint(1, 255)) for _ in range(4))
            url = f"http://{ip}/login/verify.php?user=victim&token=abc123"
        elif style == 1:
            # Typosquat
            fake = np.random.choice(typosquats)
            tld  = np.random.choice(phish_tlds)
            url = f"http://www.{fake}-secure{tld}/account/login.php"
        elif style == 2:
            # @ sign attack
            real_brand = np.random.choice(brands)
            url = f"http://secure-{real_brand}.com@evil-phisher.tk/steal"
        elif style == 3:
            # Overly long URL with many subdomains
            brand = np.random.choice(brands)
            url = f"http://secure.{brand}.verify.account.login.{brand}.update.confirm.xyz/banking/confirm"
        else:
            # Sensitive keywords + suspicious TLD
            kw = np.random.choice(["login", "verify", "secure", "banking"])
            url = f"http://{kw}-now.cf/paypal/signin.php?user=test&password=reset"
        rows.append(url)
        labels.append(1)

    df = pd.DataFrame({"url": rows, "label": labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    """Load dataset — tries real file first, falls back to synthetic."""
    if os.path.exists(DATASET_PATH):
        print(f"[INFO] Loading dataset from {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        # Try to find URL and label columns
        url_col = next((c for c in df.columns if "url" in c.lower()), None)
        lbl_col = next((c for c in df.columns if "label" in c.lower() or "type" in c.lower()), None)
        if url_col and lbl_col:
            df = df[[url_col, lbl_col]].rename(columns={url_col: "url", lbl_col: "label"})
            df["label"] = df["label"].astype(int)
            print(f"[INFO] Loaded {len(df)} samples from CSV.")
            return df.dropna().sample(min(50000, len(df)), random_state=42)
    return generate_synthetic_dataset()


def extract_features(urls: list, labels: list) -> tuple:
    """Extract all features from a list of URLs."""
    all_rows = []
    for i, url in enumerate(urls):
        if i % 500 == 0:
            print(f"  [features] {i}/{len(urls)} URLs processed...")
        try:
            url_feats = extract_url_features(url)
            typosquat = get_typosquat_score(url)
            row = {**url_feats, **CONTENT_FEATURE_DEFAULTS, "typosquat_score": typosquat}
            all_rows.append(row)
        except Exception:
            all_rows.append({f: 0 for f in ALL_FEATURE_NAMES})

    df_feats = pd.DataFrame(all_rows, columns=ALL_FEATURE_NAMES).fillna(0)
    return df_feats.values, np.array(labels)


def train():
    print("\n" + "="*60)
    print("  PhishGuard Pro — Model Training")
    print("="*60)

    # ── Load Data ────────────────────────────────────────────────────────────
    df = load_dataset()
    urls   = df["url"].tolist()
    labels = df["label"].tolist()
    print(f"[INFO] Dataset: {len(urls)} samples | "
          f"Phishing: {sum(labels)} | Legit: {len(labels)-sum(labels)}")

    # ── Feature Extraction ───────────────────────────────────────────────────
    print("[INFO] Extracting features...")
    X, y = extract_features(urls, labels)

    # ── Train/Test Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── XGBoost Model ────────────────────────────────────────────────────────
    print("[INFO] Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    # ── Save Model ───────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[SAVED] Model -> {MODEL_PATH}")

    # ── SHAP Explainer ───────────────────────────────────────────────────────
    print("[INFO] Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    with open(EXPLAINER_PATH, "wb") as f:
        pickle.dump(explainer, f)
    print(f"[SAVED] SHAP Explainer -> {EXPLAINER_PATH}")

    print("\n[DONE] Training complete! Start the API with: python run.py")


if __name__ == "__main__":
    train()
