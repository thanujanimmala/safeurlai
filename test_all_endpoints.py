import sys
import requests
import json

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"

def test_system():
    print("========================================")
    print("Testing PhishGuard Pro Backend Endpoints")
    print("========================================")

    # 1. Test stats
    r = requests.get(f"{BASE_URL}/api/stats")
    assert r.status_code == 200, f"Stats failed: {r.status_code}"
    print(f"[PASS] GET /api/stats -> {r.json()}")

    # 2. Test analyze safe URL
    payload_safe = {"url": "https://www.google.com", "fetch_content": False}
    r = requests.post(f"{BASE_URL}/api/analyze", json=payload_safe)
    assert r.status_code == 200, f"Analyze safe failed: {r.status_code}"
    data = r.json()
    print(f"[PASS] POST /api/analyze (safe: google.com)")
    print(f"       Risk Level : {data['risk_level']} ({data['risk_color']})")
    print(f"       Phish Prob : {data['phishing_probability']}%")
    print(f"       Typo-squat : {data['typosquat_score']}%")

    # 3. Test analyze typosquatted phishing URL
    payload_phish = {"url": "http://paypa1.tk/login/verify.php", "fetch_content": False}
    r = requests.post(f"{BASE_URL}/api/analyze", json=payload_phish)
    assert r.status_code == 200, f"Analyze phish failed: {r.status_code}"
    data = r.json()
    print(f"[PASS] POST /api/analyze (phish: paypa1.tk)")
    print(f"       Risk Level : {data['risk_level']} ({data['risk_color']})")
    print(f"       Phish Prob : {data['phishing_probability']}%")
    print(f"       Typo-squat : {data['typosquat_score']}%")
    print(f"       Top SHAP   : {[f['display_name'] for f in data['feature_importances'][:3]]}")
    print(f"       Explainers : {data['explanations'][:2]}")

    # 4. Test User Report submission
    payload_report = {
        "url": "http://suspicious-bank-login.xyz",
        "model_prediction": "SAFE",
        "user_label": "PHISHING",
        "phishing_prob": 12.5,
        "typosquat_score": 45.0
    }
    r = requests.post(f"{BASE_URL}/api/report", json=payload_report)
    assert r.status_code == 200, f"Report failed: {r.status_code}"
    print(f"[PASS] POST /api/report -> {r.json().get('message')}")

    # 5. Test retrieve reports
    r = requests.get(f"{BASE_URL}/api/reports?limit=5")
    assert r.status_code == 200, f"Get reports failed: {r.status_code}"
    print(f"[PASS] GET /api/reports -> Retrieved {len(r.json().get('reports', []))} reports")

    # 6. Test retrain trigger
    r = requests.post(f"{BASE_URL}/api/retrain")
    assert r.status_code == 200, f"Retrain endpoint failed: {r.status_code}"
    print(f"[PASS] POST /api/retrain -> Status: {r.json().get('status')} | Message: {r.json().get('message')}")

    print("\n========================================")
    print("ALL 6/6 SYSTEM TESTS PASSED SUCCESSFULLY!")
    print("========================================")

if __name__ == "__main__":
    test_system()
