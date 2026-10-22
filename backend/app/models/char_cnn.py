"""
Character-Level CNN (Char-CNN) — Typo-Squatting Detector
---------------------------------------------------------
Advancement #5: Learns spatial character patterns to detect brand-spoofing
techniques like paypa1.com, g00gle.com, secure-app1e.com automatically,
without relying on hardcoded keyword dictionaries.

Architecture:
  char embedding (128 dim) → Conv1D (3,4,5 kernel sizes) → GlobalMaxPool
  → Concat → Dropout → Dense → Sigmoid score
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ─── Constants ────────────────────────────────────────────────────────────────
CHARSET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".-_/: @!#$%&*+?="
)
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 = padding
VOCAB_SIZE  = len(CHARSET) + 1
MAX_LEN     = 128  # max URL characters to process

# ─── Known Top Brands (used for synthetic training data generation) ──────────
TOP_BRANDS = [
    "paypal", "google", "apple", "amazon", "microsoft", "facebook",
    "instagram", "twitter", "netflix", "spotify", "adobe", "dropbox",
    "linkedin", "youtube", "github", "yahoo", "ebay", "bank", "chase",
    "wellsfargo", "citibank", "barclays", "hsbc", "visa", "mastercard",
]


def url_to_tensor(url: str) -> torch.Tensor:
    """Convert URL string to a padded integer tensor of shape (MAX_LEN,)."""
    url = url.lower()[:MAX_LEN]
    idxs = [CHAR_TO_IDX.get(c, 0) for c in url]
    # Pad / truncate
    idxs = idxs + [0] * (MAX_LEN - len(idxs))
    return torch.tensor(idxs, dtype=torch.long)


class CharCNN(nn.Module):
    """
    Character-level CNN for typo-squatting detection.
    Returns a scalar [0,1] probability of brand spoofing.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: List[int] = [3, 4, 5],
        dropout: float = 0.4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len) long tensor
        returns : (batch,) float tensor — typo-squatting probability
        """
        emb = self.embedding(x)          # (batch, seq, embed_dim)
        emb = emb.permute(0, 2, 1)       # (batch, embed_dim, seq)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))         # (batch, num_filters, L)
            c = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # (batch, num_filters)
            pooled.append(c)

        x = torch.cat(pooled, dim=1)      # (batch, num_filters * len(kernels))
        x = self.dropout(x)
        x = self.fc(x)                    # (batch, 1)
        return torch.sigmoid(x).squeeze(-1)


# ─── Inference Helper ─────────────────────────────────────────────────────────
_char_cnn_model: CharCNN = None
_model_path = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "char_cnn.pth")


def load_char_cnn() -> CharCNN:
    """Load the trained Char-CNN from disk (or return cached instance)."""
    global _char_cnn_model
    if _char_cnn_model is not None:
        return _char_cnn_model

    model = CharCNN()
    if os.path.exists(_model_path):
        state = torch.load(_model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
    else:
        # Model not yet trained — use heuristic fallback
        model.eval()

    _char_cnn_model = model
    return model


def _heuristic_typosquat_score(url: str) -> float:
    """
    Rule-based fallback used before the Char-CNN is trained.
    Detects common letter substitutions (0→o, 1→l/i, @→a, etc.) near brand names.
    """
    url_lower = url.lower()
    score = 0.0

    for brand in TOP_BRANDS:
        if brand in url_lower:
            continue  # Actual brand presence in URL — not necessarily spoofing

        # Substitution patterns: 0→o, 1→l or i, @→a, 3→e, 4→a, 5→s
        substituted = brand
        for real, fake_variants in [
            ("o", ["0"]), ("l", ["1", "i"]), ("i", ["1"]),
            ("a", ["@", "4"]), ("e", ["3"]), ("s", ["5"]),
        ]:
            for fake in fake_variants:
                if fake in url_lower:
                    candidate = substituted.replace(real, fake)
                    if candidate in url_lower and candidate != brand:
                        score = max(score, 0.85)

        # Edit-distance heuristic: check if domain is one char off from brand
        domain_part = url_lower.split("/")[0].replace("www.", "")
        for i in range(len(brand)):
            # Deletion
            variant = brand[:i] + brand[i + 1:]
            if len(variant) > 3 and variant in domain_part:
                score = max(score, 0.70)
            # Substitution with common confusables
            for sub_char in "0O1lI":
                if sub_char.lower() == brand[i]:
                    continue
                variant = brand[:i] + sub_char + brand[i + 1:]
                if variant.lower() in domain_part:
                    score = max(score, 0.80)

    return round(min(score, 1.0), 4)


def get_typosquat_score(url: str) -> float:
    """
    Returns a [0, 1] typo-squatting probability for the URL.
    Uses the trained Char-CNN when available, else heuristic fallback.
    """
    if os.path.exists(_model_path):
        model = load_char_cnn()
        tensor = url_to_tensor(url).unsqueeze(0)  # (1, MAX_LEN)
        with torch.no_grad():
            score = model(tensor).item()
        return round(score, 4)
    else:
        return _heuristic_typosquat_score(url)
