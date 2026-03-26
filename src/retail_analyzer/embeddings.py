"""Shared sentence-transformers encoder for text similarity."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None


def get_text_encoder() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = get_text_encoder()
    return np.asarray(
        model.encode(texts, batch_size=batch_size, show_progress_bar=False),
        dtype=np.float32,
    )
