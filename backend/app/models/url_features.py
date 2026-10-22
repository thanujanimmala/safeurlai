"""
URL Feature Extractor
---------------------
Extracts 22+ lexical features from a raw URL string.
Extends the base paper (ICICS 2021) feature set with additional entropy,
subdomain depth, and brand-proximity features.
"""

import re
import math
from urllib.parse import urlparse
from typing import Dict, Any

# ─── Known Suspicious TLDs ───────────────────────────────────────────────────
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".online", ".site", ".website", ".info", ".biz", ".mobi",
}

# ─── Sensitive Keywords (used only for count, NOT as a classifier) ──────────
SENSITIVE_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "account", "banking",
    "secure", "update", "confirm", "paypal", "ebay", "amazon",
    "apple", "google", "microsoft", "password", "credential",
    "wallet", "auth", "oauth", "reset", "validate",
]


def _char_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((count / n) * math.log2(count / n) for count in freq.values())


def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extract 30 lexical + structural features from a URL.

    Returns
    -------
    dict  — feature_name → numeric value
    """
    url = url.strip()

    # ── Parse URL ────────────────────────────────────────────────────────────
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
    except Exception:
        parsed = urlparse("")

    scheme   = parsed.scheme or ""
    hostname = parsed.hostname or ""
    path     = parsed.path or ""
    query    = parsed.query or ""
    fragment = parsed.fragment or ""

    full_url     = url
    domain       = hostname
    path_str     = path + ("?" + query if query else "")

    # ── Tokenise ─────────────────────────────────────────────────────────────
    url_tokens    = re.split(r"[\W_]+", full_url)
    domain_tokens = re.split(r"[\W_]+", domain)
    path_tokens   = re.split(r"[\W_]+", path_str)

    url_tokens    = [t for t in url_tokens    if t]
    domain_tokens = [t for t in domain_tokens if t]
    path_tokens   = [t for t in path_tokens   if t]

    # ── Feature 1–3: Lengths ─────────────────────────────────────────────────
    url_length      = len(full_url)
    hostname_length = len(hostname)
    path_length     = len(path)

    # ── Feature 4: Dot Count ─────────────────────────────────────────────────
    dots_in_url = full_url.count(".")

    # ── Feature 5–7: Token Counts ─────────────────────────────────────────────
    url_token_count    = len(url_tokens)
    domain_token_count = len(domain_tokens)
    path_token_count   = len(path_tokens)

    # ── Feature 8–10: Avg Token Lengths ─────────────────────────────────────
    def avg_len(tokens):
        return sum(len(t) for t in tokens) / len(tokens) if tokens else 0

    url_avg_token_len    = avg_len(url_tokens)
    domain_avg_token_len = avg_len(domain_tokens)
    path_avg_token_len   = avg_len(path_tokens)

    # ── Feature 11–13: Largest Token Lengths ─────────────────────────────────
    url_max_token_len    = max((len(t) for t in url_tokens),    default=0)
    domain_max_token_len = max((len(t) for t in domain_tokens), default=0)
    path_max_token_len   = max((len(t) for t in path_tokens),   default=0)

    # ── Feature 14: Sensitive Keyword Count ──────────────────────────────────
    url_lower = full_url.lower()
    sensitive_token_count = sum(kw in url_lower for kw in SENSITIVE_KEYWORDS)

    # ── Feature 15: IP Address in URL ────────────────────────────────────────
    ip_pattern = re.compile(
        r"(\d{1,3}\.){3}\d{1,3}"
        r"|0x[0-9a-fA-F]+"      # hex IP
        r"|(\d{8,10})"          # decimal IP
    )
    has_ip_address = int(bool(ip_pattern.search(hostname)))

    # ── Feature 16: @ Sign ───────────────────────────────────────────────────
    has_at_sign = int("@" in full_url)

    # ── Feature 17: URL Depth (forward slashes in path) ──────────────────────
    url_depth = path.count("/")

    # ── Feature 18: URL Redirection (//) ─────────────────────────────────────
    has_double_slash = int("//" in path)

    # ── Feature 19: HTTPS ────────────────────────────────────────────────────
    has_https = int(scheme == "https")

    # ── Feature 20: Subdomain count ──────────────────────────────────────────
    parts = hostname.split(".")
    subdomain_count = max(len(parts) - 2, 0) if len(parts) > 2 else 0

    # ── Feature 21: Hyphen in Domain ─────────────────────────────────────────
    hyphen_count = domain.count("-")

    # ── Feature 22: Suspicious TLD ───────────────────────────────────────────
    tld = "." + parts[-1].lower() if parts else ""
    has_suspicious_tld = int(tld in SUSPICIOUS_TLDS)

    # ── Feature 23: Special Char Count ───────────────────────────────────────
    special_chars = re.findall(r"[^a-zA-Z0-9\-._/:]", full_url)
    special_char_count = len(special_chars)

    # ── Feature 24: URL Entropy ───────────────────────────────────────────────
    url_entropy = _char_entropy(full_url)

    # ── Feature 25: Domain Entropy ────────────────────────────────────────────
    domain_entropy = _char_entropy(domain)

    # ── Feature 26: Digit Ratio in Domain ────────────────────────────────────
    digit_ratio_in_domain = (
        sum(c.isdigit() for c in domain) / len(domain) if domain else 0
    )

    # ── Feature 27: Has Query String ─────────────────────────────────────────
    has_query = int(bool(query))

    # ── Feature 28: Query Length ─────────────────────────────────────────────
    query_length = len(query)

    # ── Feature 29: Fragment in URL ──────────────────────────────────────────
    has_fragment = int(bool(fragment))

    # ── Feature 30: Punycode / IDN (homograph attack) ────────────────────────
    has_punycode = int("xn--" in hostname.lower())

    features = {
        "url_length":            url_length,
        "hostname_length":       hostname_length,
        "path_length":           path_length,
        "dots_in_url":           dots_in_url,
        "url_token_count":       url_token_count,
        "domain_token_count":    domain_token_count,
        "path_token_count":      path_token_count,
        "url_avg_token_len":     url_avg_token_len,
        "domain_avg_token_len":  domain_avg_token_len,
        "path_avg_token_len":    path_avg_token_len,
        "url_max_token_len":     url_max_token_len,
        "domain_max_token_len":  domain_max_token_len,
        "path_max_token_len":    path_max_token_len,
        "sensitive_token_count": sensitive_token_count,
        "has_ip_address":        has_ip_address,
        "has_at_sign":           has_at_sign,
        "url_depth":             url_depth,
        "has_double_slash":      has_double_slash,
        "has_https":             has_https,
        "subdomain_count":       subdomain_count,
        "hyphen_count":          hyphen_count,
        "has_suspicious_tld":    has_suspicious_tld,
        "special_char_count":    special_char_count,
        "url_entropy":           url_entropy,
        "domain_entropy":        domain_entropy,
        "digit_ratio_in_domain": digit_ratio_in_domain,
        "has_query":             has_query,
        "query_length":          query_length,
        "has_fragment":          has_fragment,
        "has_punycode":          has_punycode,
    }

    return features


FEATURE_NAMES = list(extract_url_features("https://example.com").keys())
