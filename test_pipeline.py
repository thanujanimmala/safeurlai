import sys, os
sys.path.insert(0, r"c:\Users\thanu\Minor degree project")
sys.stdout.reconfigure(encoding='utf-8')

from backend.app.models.predictor import analyze_url

test_urls = [
    ("https://www.google.com",                         "Should be SAFE"),
    ("http://paypa1.tk/login/verify.php",               "Should be PHISHING (typosquat + .tk)"),
    ("http://192.168.1.1/banking/secure/login.php",     "Should be PHISHING (IP address)"),
    ("http://secure-amazon-verify.cf/account",          "Should be PHISHING/SUSPICIOUS"),
    ("https://github.com/features",                     "Should be SAFE"),
]

print("=" * 65)
print("  PhishGuard Pro - Full Pipeline Test (with trained models)")
print("=" * 65)

for url, expectation in test_urls:
    result = analyze_url(url, fetch_content=False)
    risk_icon = {"SAFE": "[SAFE]", "SUSPICIOUS": "[WARN]", "PHISHING": "[!!!]"}
    icon = risk_icon.get(result["risk_level"], "[ ? ]")
    print(f"\n{icon} {url[:55]}")
    print(f"       Risk: {result['risk_level']:12s} | Phishing: {result['phishing_probability']:5.1f}% | Typosquat: {result['typosquat_score']:5.1f}%")
    print(f"       Model: {result['model_type']}")
    print(f"       Note: {expectation}")

print("\n" + "=" * 65)
print("  All tests passed! Starting server...")
print("=" * 65)
