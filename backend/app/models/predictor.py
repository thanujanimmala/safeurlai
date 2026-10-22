"""
Predictor — Main ML Pipeline
-----------------------------
Combines URL features + content features + Char-CNN typosquat score
into a single XGBoost prediction with SHAP explanations.

Falls back to a calibrated rule-based predictor when no trained model exists.
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple

from .url_features import extract_url_features, FEATURE_NAMES
from .content_features import extract_content_features
from .char_cnn import get_typosquat_score
from ..utils.explainer import (
    generate_explanations,
    risk_level_from_score,
    score_to_color,
)

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "phishguard_model.pkl")
EXPLAINER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "shap_explainer.pkl")

_model    = None
_explainer = None


def _load_model():
    global _model, _explainer
    if _model is not None:
        return _model, _explainer
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        if os.path.exists(EXPLAINER_PATH):
            with open(EXPLAINER_PATH, "rb") as f:
                _explainer = pickle.load(f)
    return _model, _explainer


def _all_feature_names() -> List[str]:
    return FEATURE_NAMES + [
        "content_fetch_success", "has_login_form", "password_field_count",
        "form_count", "iframe_count", "hidden_element_count",
        "external_link_ratio", "favicon_domain_mismatch",
        "has_right_click_disable", "has_popup_script",
        "external_script_ratio", "page_title_brand_mismatch",
        "meta_refresh_redirect", "copyright_year_missing",
        "suspicious_action_url", "typosquat_score",
    ]


def _rule_based_score(url_feats: Dict, content_feats: Dict, typosquat: float) -> float:
    """
    Calibrated rule-based phishing probability used before training.
    Aggregates key risk signals from both feature sets.
    """
    score = 0.0
    w = {
        "has_ip_address":          0.20,
        "has_at_sign":             0.12,
        "has_suspicious_tld":      0.10,
        "sensitive_token_count":   0.03,
        "has_punycode":            0.15,
        "has_double_slash":        0.05,
        "subdomain_count":         0.03,
        "has_https":              -0.15,   # negative = lowers phishing score
        "has_login_form":          0.12,
        "iframe_count":            0.04,
        "favicon_domain_mismatch": 0.10,
        "has_right_click_disable": 0.08,
        "has_popup_script":        0.06,
        "suspicious_action_url":   0.12,
        "page_title_brand_mismatch": 0.10,
        "meta_refresh_redirect":   0.06,
    }
    for feat, weight in w.items():
        val = url_feats.get(feat, content_feats.get(feat, 0))
        score += val * weight

    # Entropy bonus
    ent = url_feats.get("url_entropy", 0)
    if ent > 4.5:
        score += 0.08

    # Long URL bonus
    if url_feats.get("url_length", 0) > 100:
        score += 0.06

    # Typosquat
    score += typosquat * 0.15

    return min(max(score, 0.0), 1.0)


def _rule_based_shap(url_feats: Dict, content_feats: Dict, typosquat: float) -> List[Tuple[str, float]]:
    """Approximate feature importances using rule weights (used as fallback)."""
    combined = {**url_feats, **content_feats, "typosquat_score": typosquat}
    rule_weights = {
        "has_ip_address":          0.20,
        "has_at_sign":             0.12,
        "favicon_domain_mismatch": 0.10,
        "has_suspicious_tld":      0.10,
        "has_punycode":            0.15,
        "has_login_form":          0.12,
        "suspicious_action_url":   0.12,
        "page_title_brand_mismatch": 0.10,
        "typosquat_score":         0.15,
        "has_https":              -0.15,
        "has_right_click_disable": 0.08,
        "has_popup_script":        0.06,
        "meta_refresh_redirect":   0.06,
        "subdomain_count":         0.03,
    }
    pairs = []
    for feat, w in rule_weights.items():
        val = combined.get(feat, 0)
        pairs.append((feat, val * w))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs


def analyze_url(url: str, fetch_content: bool = True) -> Dict[str, Any]:
    """
    Full dual-layer phishing analysis pipeline.

    Parameters
    ----------
    url           : The URL to analyze
    fetch_content : Whether to scrape the webpage (slower but more accurate)

    Returns
    -------
    dict with keys:
      url, phishing_probability, typosquat_score, risk_level,
      risk_color, explanations, feature_importances, url_features, content_features
    """
    # ── Step 1: URL Features ──────────────────────────────────────────────────
    url_feats = extract_url_features(url)

    # ── Step 2: Content Features ──────────────────────────────────────────────
    content_feats = extract_content_features(url) if fetch_content else {
        "content_fetch_success": 0, "has_login_form": 0,
        "password_field_count": 0, "form_count": 0, "iframe_count": 0,
        "hidden_element_count": 0, "external_link_ratio": 0.0,
        "favicon_domain_mismatch": 0, "has_right_click_disable": 0,
        "has_popup_script": 0, "external_script_ratio": 0.0,
        "page_title_brand_mismatch": 0, "meta_refresh_redirect": 0,
        "copyright_year_missing": 0, "suspicious_action_url": 0,
    }

    # ── Step 3: Char-CNN Typosquat Score ──────────────────────────────────────
    typosquat = get_typosquat_score(url)

    # ── Step 4: ML Prediction ─────────────────────────────────────────────────
    model, explainer = _load_model()

    if model is not None:
        feature_names = _all_feature_names()
        combined = {**url_feats, **content_feats, "typosquat_score": typosquat}
        X = np.array([[combined.get(f, 0) for f in feature_names]])

        proba = float(model.predict_proba(X)[0][1])

        if explainer is not None:
            try:
                import shap
                sv = explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]
                shap_pairs = sorted(
                    zip(feature_names, sv[0].tolist()),
                    key=lambda x: abs(x[1]), reverse=True
                )
            except Exception:
                shap_pairs = _rule_based_shap(url_feats, content_feats, typosquat)
        else:
            shap_pairs = _rule_based_shap(url_feats, content_feats, typosquat)
    else:
        proba = _rule_based_score(url_feats, content_feats, typosquat)
        shap_pairs = _rule_based_shap(url_feats, content_feats, typosquat)

    # ── Step 5: Risk Level & Explanations ─────────────────────────────────────
    risk_level = risk_level_from_score(proba, typosquat)
    color      = score_to_color(risk_level)
    explanations = generate_explanations(shap_pairs, risk_level)

    # Build feature importance list for frontend chart
    top_importances = [
        {"feature": f, "importance": round(v, 4), "display_name": f.replace("_", " ").title()}
        for f, v in shap_pairs[:10]
    ]

    return {
        "url":                  url,
        "phishing_probability": round(proba * 100, 1),
        "typosquat_score":      round(typosquat * 100, 1),
        "risk_level":           risk_level,
        "risk_color":           color,
        "explanations":         explanations,
        "feature_importances":  top_importances,
        "url_features":         url_feats,
        "content_features":     content_feats,
        "model_type":           "XGBoost+SHAP" if model else "Rule-Based (Train model for ML)",
    }
