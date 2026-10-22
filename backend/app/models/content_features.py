"""
Webpage Content Feature Extractor
----------------------------------
Advancement #2: Fetches live webpage HTML and extracts DOM-level
phishing indicators that URL-only approaches completely miss.
"""

import re
import time
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional


FETCH_TIMEOUT = 8   # seconds
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _safe_get(url: str) -> Optional[str]:
    """Fetch URL HTML safely. Returns None on failure."""
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers=HEADERS,
                            allow_redirects=True, verify=False)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def extract_content_features(url: str) -> Dict[str, Any]:
    """
    Fetch the webpage at `url` and extract 15 DOM-level phishing features.

    Returns a dict of features + a `content_fetch_success` flag.
    If the page is unreachable, all features default to 0 (neutral).
    """
    default = {
        "content_fetch_success":   0,
        "has_login_form":          0,
        "password_field_count":    0,
        "form_count":              0,
        "iframe_count":            0,
        "hidden_element_count":    0,
        "external_link_ratio":     0.0,
        "favicon_domain_mismatch": 0,
        "has_right_click_disable": 0,
        "has_popup_script":        0,
        "external_script_ratio":   0.0,
        "page_title_brand_mismatch": 0,
        "meta_refresh_redirect":   0,
        "copyright_year_missing":  0,
        "suspicious_action_url":   0,
    }

    html = _safe_get(url)
    if html is None:
        return default

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return default

    parsed_url = urlparse(url)
    base_domain = parsed_url.netloc.lower()

    # ── content_fetch_success ────────────────────────────────────────────────
    default["content_fetch_success"] = 1

    # ── Forms & Login ────────────────────────────────────────────────────────
    forms = soup.find_all("form")
    form_count = len(forms)
    default["form_count"] = form_count

    password_fields = soup.find_all("input", {"type": "password"})
    default["password_field_count"] = len(password_fields)
    default["has_login_form"] = int(len(password_fields) > 0)

    # ── Iframes ──────────────────────────────────────────────────────────────
    iframes = soup.find_all("iframe")
    default["iframe_count"] = len(iframes)

    # ── Hidden Elements ───────────────────────────────────────────────────────
    hidden = soup.find_all(
        lambda tag: tag.get("style") and "display:none" in tag.get("style", "").replace(" ", "")
        or tag.get("hidden") is not None
        or tag.get("type", "").lower() == "hidden"
    )
    default["hidden_element_count"] = len(hidden)

    # ── External Links Ratio ─────────────────────────────────────────────────
    all_links = soup.find_all("a", href=True)
    external_links = [
        a for a in all_links
        if urlparse(a["href"]).netloc not in ("", base_domain)
        and a["href"].startswith("http")
    ]
    default["external_link_ratio"] = (
        len(external_links) / len(all_links) if all_links else 0.0
    )

    # ── Favicon domain mismatch ───────────────────────────────────────────────
    favicon_tag = (
        soup.find("link", rel=lambda r: r and "icon" in r)
        or soup.find("link", rel="shortcut icon")
    )
    if favicon_tag and favicon_tag.get("href"):
        favicon_href = favicon_tag["href"]
        if favicon_href.startswith("http"):
            favicon_domain = urlparse(favicon_href).netloc.lower()
            default["favicon_domain_mismatch"] = int(
                favicon_domain not in ("", base_domain)
            )

    # ── JavaScript Tricks ────────────────────────────────────────────────────
    page_text = html.lower()
    default["has_right_click_disable"] = int(
        "contextmenu" in page_text or "event.button==2" in page_text
    )
    default["has_popup_script"] = int(
        "window.open" in page_text or "showmodaldialog" in page_text
    )

    # ── External Script Ratio ─────────────────────────────────────────────────
    scripts = soup.find_all("script", src=True)
    ext_scripts = [
        s for s in scripts
        if urlparse(s["src"]).netloc not in ("", base_domain)
    ]
    default["external_script_ratio"] = (
        len(ext_scripts) / len(scripts) if scripts else 0.0
    )

    # ── Page Title vs Domain Brand Mismatch ───────────────────────────────────
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        title_lower = title_tag.string.lower()
        # Check if major brand names appear in title but not in domain
        major_brands = ["paypal", "apple", "google", "amazon", "microsoft",
                        "facebook", "bank", "chase", "wellsfargo", "netflix"]
        for brand in major_brands:
            if brand in title_lower and brand not in base_domain:
                default["page_title_brand_mismatch"] = 1
                break

    # ── Meta Refresh Redirect ─────────────────────────────────────────────────
    meta_refresh = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
    default["meta_refresh_redirect"] = int(meta_refresh is not None)

    # ── Copyright year missing ────────────────────────────────────────────────
    current_year = time.localtime().tm_year
    has_copyright = bool(
        re.search(r"©|&copy;|copyright", page_text, re.I)
        and str(current_year) in page_text
    )
    default["copyright_year_missing"] = int(not has_copyright)

    # ── Suspicious Form Action URL ────────────────────────────────────────────
    for form in forms:
        action = form.get("action", "")
        if action.startswith("http"):
            action_domain = urlparse(action).netloc.lower()
            if action_domain and action_domain != base_domain:
                default["suspicious_action_url"] = 1
                break

    return default


CONTENT_FEATURE_NAMES = list(extract_content_features.__doc__ and
                              [k for k in [
                                  "content_fetch_success", "has_login_form",
                                  "password_field_count", "form_count",
                                  "iframe_count", "hidden_element_count",
                                  "external_link_ratio", "favicon_domain_mismatch",
                                  "has_right_click_disable", "has_popup_script",
                                  "external_script_ratio", "page_title_brand_mismatch",
                                  "meta_refresh_redirect", "copyright_year_missing",
                                  "suspicious_action_url",
                              ]])
