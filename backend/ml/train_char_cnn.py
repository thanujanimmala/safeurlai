# -*- coding: utf-8 -*-
"""
Char-CNN Training Script — Typo-Squatting Detector
====================================================
Trains the Character-level CNN to detect brand name typo-squatting.

Generates synthetic training data from top 25 brand names by creating
realistic typo-squatted variants using character substitution rules.

Usage:
    python backend/ml/train_char_cnn.py
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.app.models.char_cnn import (
    CharCNN, url_to_tensor, TOP_BRANDS, MAX_LEN, _model_path
)

EPOCHS     = 15
BATCH_SIZE = 256
LR         = 1e-3
SAVE_PATH  = _model_path


# ─── Substitution Rules for Synthetic Typosquats ─────────────────────────────
SUBSTITUTIONS = {
    "a": ["@", "4", "а"],   # last is Cyrillic а
    "e": ["3", "е"],         # Cyrillic е
    "i": ["1", "l", "і"],   # Cyrillic і
    "l": ["1", "I", "i"],
    "o": ["0", "о"],         # Cyrillic о
    "s": ["5", "$"],
    "t": ["7"],
    "g": ["9"],
    "b": ["6"],
}

TOP_LEGIT_DOMAINS = [
    f"https://www.{brand}.com/{path}"
    for brand in TOP_BRANDS
    for path in ["login", "account", "home", "", "profile", "settings"]
]

RANDOM_LEGIT = [
    f"https://www.{word}.com"
    for word in [
        "techcrunch", "medium", "stackoverflow", "reddit", "bbc",
        "nytimes", "openai", "pytorch", "fastapi", "wikipedia",
        "mozilla", "python", "numpy", "github", "gitlab",
    ]
]


def make_typosquat(brand: str) -> str:
    """Generate a realistic typo-squatted URL for a brand."""
    brand_lower = brand.lower()
    # Choose a random position to substitute
    for _ in range(20):
        idx = random.randint(0, len(brand_lower) - 1)
        char = brand_lower[idx]
        if char in SUBSTITUTIONS:
            fake_char = random.choice(SUBSTITUTIONS[char])
            fake_brand = brand_lower[:idx] + fake_char + brand_lower[idx + 1:]
            # Build a plausible phishing URL
            suffix = random.choice([
                "-secure.tk", "-login.ml", "-verify.cf",
                "-account.ga", "-banking.xyz", ".phish.club",
            ])
            path = random.choice([
                "/login.php", "/verify.php?user=victim",
                "/account/signin", "/confirm?token=xyz123",
            ])
            return f"http://www.{fake_brand}{suffix}{path}"
    # Fallback: add suspicious suffix
    return f"http://www.{brand_lower}-secure.tk/login.php"


def generate_training_data(n_per_brand: int = 200):
    """Generate (url, label) pairs: 1 = typosquat, 0 = legitimate."""
    data = []

    # Positive samples (typosquats)
    for brand in TOP_BRANDS:
        for _ in range(n_per_brand):
            url = make_typosquat(brand)
            data.append((url, 1))

    # Negative samples (legitimate URLs)
    for url in TOP_LEGIT_DOMAINS:
        data.append((url, 0))
    for url in RANDOM_LEGIT:
        data.append((url, 0))

    # Extra negatives to balance
    while sum(1 for _, l in data if l == 0) < sum(1 for _, l in data if l == 1):
        brand = random.choice(TOP_BRANDS)
        url = f"https://www.{brand}.com/{random.choice(['home', 'contact', 'about'])}"
        data.append((url, 0))

    random.shuffle(data)
    return data


class URLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        url, label = self.data[idx]
        return url_to_tensor(url), torch.tensor(label, dtype=torch.float32)


def train():
    print("\n" + "="*60)
    print("  PhishGuard Pro — Char-CNN Training (Typo-Squatting)")
    print("="*60)

    data = generate_training_data(n_per_brand=300)
    print(f"[INFO] Dataset: {len(data)} samples | "
          f"Typosquat: {sum(1 for _, l in data if l==1)} | "
          f"Legit: {sum(1 for _, l in data if l==0)}")

    # Split
    split = int(0.85 * len(data))
    train_data = data[:split]
    test_data  = data[split:]

    train_loader = DataLoader(URLDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(URLDataset(test_data),  batch_size=BATCH_SIZE)

    model = CharCNN()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X)
                predicted = (pred > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"[Epoch {epoch+1:02d}/{EPOCHS}] Loss={total_loss/len(train_loader):.4f}  Test Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  [SAVED] Best model (acc={best_acc:.4f})")

    print(f"\n[DONE] Char-CNN training complete! Best accuracy: {best_acc:.4f}")
    print(f"   Model saved -> {SAVE_PATH}")


if __name__ == "__main__":
    train()
