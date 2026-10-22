"""
Active Learning Retraining Pipeline
=====================================
Advancement #4: Picks up user-reported URLs from the database,
verifies them (optionally via VirusTotal), and warm-start retrains
the XGBoost model to adapt to emerging phishing campaigns.

Usage:
    python backend/ml/retrain.py          # manual trigger
    # Or triggered automatically by POST /api/retrain
"""

import os
import sys
import pickle
import sqlite3
import numpy as np
import xgboost as xgb
import shap
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.app.models.url_features import extract_url_features, FEATURE_NAMES
from backend.app.models.char_cnn import get_typosquat_score
from backend.app.database.db import (
    get_pending_reports, mark_reports_retrained, DB_PATH, get_connection
)

MODEL_DIR      = os.path.dirname(__file__)
MODEL_PATH     = os.path.join(MODEL_DIR, "phishguard_model.pkl")
EXPLAINER_PATH = os.path.join(MODEL_DIR, "shap_explainer.pkl")

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

MIN_REPORTS_FOR_RETRAIN = 5


def verify_with_virustotal(url: str, api_key: str = "") -> int:
    """
    Verify a URL with VirusTotal.
    Returns 1 if phishing confirmed, 0 if safe, -1 if unknown/error.
    Set VIRUSTOTAL_API_KEY env variable for real verification.
    """
    if not api_key:
        return -1  # Skip if no API key
    try:
        import requests, hashlib, base64
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        resp = requests.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}",
            headers={"x-apikey": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            stats = resp.json()["data"]["attributes"]["last_analysis_stats"]
            if stats.get("malicious", 0) >= 3:
                return 1
            elif stats.get("clean", 0) > 10:
                return 0
    except Exception:
        pass
    return -1


def url_to_feature_row(url: str) -> np.ndarray:
    url_feats = extract_url_features(url)
    typosquat = get_typosquat_score(url)
    combined  = {**url_feats, **CONTENT_FEATURE_DEFAULTS, "typosquat_score": typosquat}
    return np.array([[combined.get(f, 0) for f in ALL_FEATURE_NAMES]])


def retrain():
    print("\n" + "="*60)
    print(f"  PhishGuard Pro — Active Learning Retrain ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("="*60)

    reports = get_pending_reports(limit=500)
    if len(reports) < MIN_REPORTS_FOR_RETRAIN:
        print(f"[SKIP] Only {len(reports)} pending reports. Need ≥ {MIN_REPORTS_FOR_RETRAIN}.")
        return

    print(f"[INFO] {len(reports)} user reports found.")

    # ── Load existing model ───────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] No trained model found. Run train_model.py first.")
        return

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # ── Build new training set from reports ──────────────────────────────────
    vt_key = os.getenv("VIRUSTOTAL_API_KEY", "")
    X_new, y_new, used_ids = [], [], []

    for report in reports:
        url   = report["url"]
        label = 1 if report["user_label"].upper() == "PHISHING" else 0

        # Optional VirusTotal verification
        vt_result = verify_with_virustotal(url, vt_key)
        if vt_result != -1:
            label = vt_result  # Override with verified label

        try:
            row = url_to_feature_row(url)
            X_new.append(row[0])
            y_new.append(label)
            used_ids.append(report["id"])
        except Exception as e:
            print(f"[WARN] Skipping {url}: {e}")

    if not X_new:
        print("[WARN] No valid feature rows extracted.")
        return

    X_new = np.array(X_new)
    y_new = np.array(y_new)
    print(f"[INFO] New training samples: {len(X_new)} | "
          f"Phishing: {y_new.sum()} | Safe: {(y_new==0).sum()}")

    # ── Warm-start retrain: update existing model with new data ──────────────
    try:
        acc_before = _quick_eval(model, X_new, y_new)
        print(f"[INFO] Model accuracy on new data BEFORE retrain: {acc_before:.4f}")

        model.fit(
            X_new, y_new,
            xgb_model=model.get_booster(),  # warm-start from existing model
            eval_metric="logloss",
            verbose=False,
        )

        acc_after = _quick_eval(model, X_new, y_new)
        print(f"[INFO] Model accuracy on new data AFTER retrain:  {acc_after:.4f}")

        # Save updated model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"[SAVED] Updated model → {MODEL_PATH}")

        # Regenerate SHAP explainer
        explainer = shap.TreeExplainer(model)
        with open(EXPLAINER_PATH, "wb") as f:
            pickle.dump(explainer, f)
        print(f"[SAVED] SHAP Explainer → {EXPLAINER_PATH}")

        # Mark reports as used
        mark_reports_retrained(used_ids)

        # Log retrain event
        conn = get_connection()
        conn.execute("""
            INSERT INTO retrain_log (samples_used, accuracy_before, accuracy_after, created_at)
            VALUES (?, ?, ?, ?)
        """, (len(used_ids), float(acc_before), float(acc_after), datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

        print(f"\n✅ Retraining complete! Used {len(used_ids)} user reports.")

    except Exception as e:
        print(f"[ERROR] Retraining failed: {e}")
        raise


def _quick_eval(model, X, y) -> float:
    """Quick accuracy evaluation."""
    preds = model.predict(X)
    return (preds == y).mean()


if __name__ == "__main__":
    retrain()
