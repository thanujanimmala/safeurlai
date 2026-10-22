"""
SHAP Explainer
--------------
Advancement #3: Generates human-readable explanations for why the model
classified a URL as Safe / Suspicious / Phishing.

Uses TreeSHAP for the XGBoost model — runs in < 50ms.
"""

from typing import List, Dict, Any, Tuple

# ─── Human-readable feature descriptions ──────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    # URL lexical features
    "url_length":              ("URL is very long",                 "URL is normal length"),
    "hostname_length":         ("Hostname is unusually long",        "Hostname length is normal"),
    "path_length":             ("URL path is very long",            "URL path length is normal"),
    "dots_in_url":             ("Too many dots in the URL",         "Normal dot count"),
    "url_token_count":         ("URL has many unusual tokens",      "Normal URL structure"),
    "domain_token_count":      ("Domain has many tokens",           "Normal domain structure"),
    "path_token_count":        ("Path has many segments",           "Normal path structure"),
    "url_avg_token_len":       ("Tokens are unusually short/long",  "Normal token lengths"),
    "domain_avg_token_len":    ("Domain tokens are suspicious",     "Domain structure is normal"),
    "path_avg_token_len":      ("Path tokens are suspicious",       "Normal path tokens"),
    "url_max_token_len":       ("Contains very long suspicious token", "Token lengths are normal"),
    "domain_max_token_len":    ("Domain has very long token",       "Domain token length is normal"),
    "path_max_token_len":      ("Path has very long token",         "Normal path token length"),
    "sensitive_token_count":   ("Contains sensitive keywords (login/bank)", "No suspicious keywords"),
    "has_ip_address":          ("Uses IP address instead of domain name", "Uses proper domain name"),
    "has_at_sign":             ("Contains @ sign — hides real destination", "No @ sign tricks"),
    "url_depth":               ("URL is deeply nested (many slashes)", "Normal URL depth"),
    "has_double_slash":        ("Contains //, often used to redirect", "No redirect tricks"),
    "has_https":               ("No HTTPS (not encrypted)",         "Uses HTTPS (encrypted)"),
    "subdomain_count":         ("Excessive subdomains",             "Normal subdomain count"),
    "hyphen_count":            ("Many hyphens — brand mimicry tactic", "Normal hyphen usage"),
    "has_suspicious_tld":      ("Uses suspicious domain extension (.tk/.ml/etc)", "Normal domain extension"),
    "special_char_count":      ("Many special characters",         "Normal characters"),
    "url_entropy":             ("URL appears randomly generated",   "URL looks human-readable"),
    "domain_entropy":          ("Domain appears random/generated",  "Domain looks legitimate"),
    "digit_ratio_in_domain":   ("Domain has many digits — suspicious", "Normal domain characters"),
    "has_query":               ("Has query parameters",            "No suspicious query"),
    "query_length":            ("Very long query string",          "Normal query length"),
    "has_fragment":            ("Has fragment identifier",         "No suspicious fragment"),
    "has_punycode":            ("Uses Punycode (homograph attack)", "Normal domain encoding"),
    # Content features
    "content_fetch_success":   ("Page could not be fetched",       "Page fetched successfully"),
    "has_login_form":          ("Has login/password form",         "No login form found"),
    "password_field_count":    ("Has password input fields",       "No password fields"),
    "form_count":              ("Multiple suspicious forms",       "Normal form count"),
    "iframe_count":            ("Contains hidden iframes",         "No suspicious iframes"),
    "hidden_element_count":    ("Contains hidden elements",        "No hidden elements"),
    "external_link_ratio":     ("Mostly links to external sites",  "Links are internal"),
    "favicon_domain_mismatch": ("Favicon from different domain",   "Favicon matches domain"),
    "has_right_click_disable": ("Right-click is disabled",         "Right-click works normally"),
    "has_popup_script":        ("Has popup/redirect scripts",      "No suspicious scripts"),
    "external_script_ratio":   ("Scripts loaded from external sites", "Scripts are local"),
    "page_title_brand_mismatch": ("Page title mentions different brand", "Brand matches domain"),
    "meta_refresh_redirect":   ("Auto-redirects the page",        "No auto-redirect"),
    "copyright_year_missing":  ("Copyright year is missing",       "Copyright year is present"),
    "suspicious_action_url":   ("Form submits to different domain", "Form submits locally"),
    # Char-CNN
    "typosquat_score":         ("Looks like brand name with typos", "No typo-squatting detected"),
}


def generate_explanations(
    feature_importances: List[Tuple[str, float]],
    risk_level: str,
) -> List[str]:
    """
    Convert SHAP feature importance values into human-readable sentences.

    Parameters
    ----------
    feature_importances : list of (feature_name, shap_value) — sorted by |shap_value|
    risk_level          : "SAFE" | "SUSPICIOUS" | "PHISHING"

    Returns
    -------
    list[str] — top 5 explanations in plain English
    """
    explanations = []
    for feat_name, shap_val in feature_importances[:5]:
        if feat_name not in FEATURE_DESCRIPTIONS:
            continue
        positive_desc, negative_desc = FEATURE_DESCRIPTIONS[feat_name]
        # Positive SHAP = pushes toward phishing; negative = pushes toward safe
        if shap_val > 0:
            explanations.append(f"⚠️ {positive_desc}")
        else:
            explanations.append(f"✅ {negative_desc}")

    if not explanations:
        if risk_level == "SAFE":
            explanations = ["✅ URL structure appears legitimate",
                            "✅ No suspicious indicators found"]
        else:
            explanations = ["⚠️ URL shows multiple suspicious patterns"]

    return explanations


def risk_level_from_score(phishing_prob: float, typosquat_score: float) -> str:
    """
    Convert raw phishing probability + typosquat score into a 3-tier risk level.

    Tiers:
      🟢 SAFE        — prob < 0.35 and typosquat < 0.5
      🟡 SUSPICIOUS  — prob in [0.35, 0.65] OR typosquat >= 0.5
      🔴 PHISHING    — prob >= 0.65
    """
    if phishing_prob >= 0.65:
        return "PHISHING"
    elif phishing_prob >= 0.35 or typosquat_score >= 0.5:
        return "SUSPICIOUS"
    else:
        return "SAFE"


def score_to_color(risk_level: str) -> str:
    return {
        "SAFE":       "#10b981",
        "SUSPICIOUS": "#f59e0b",
        "PHISHING":   "#ef4444",
    }.get(risk_level, "#6b7280")
