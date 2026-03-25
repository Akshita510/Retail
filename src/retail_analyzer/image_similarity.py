from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from retail_analyzer.config import AnalyzerConfig
from retail_analyzer.excel_io import parse_image_urls
from retail_analyzer.image_fetch import cached_fetch

logger = logging.getLogger(__name__)

_MODEL: SentenceTransformer | None = None


def get_clip_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        # Small, fast, good for "same product, different pose".
        _MODEL = SentenceTransformer("clip-ViT-B-32")
    return _MODEL


def _encode_url_batch(
    urls: list[str],
    config: AnalyzerConfig,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Fetch + embed each URL once. Omitted keys failed download/encode."""
    if not urls:
        return {}
    model = get_clip_model()
    url_to_vec: dict[str, np.ndarray] = {}
    images_batch: list = []
    urls_batch: list[str] = []

    def flush() -> None:
        nonlocal images_batch, urls_batch
        if not images_batch:
            return
        embs = model.encode(
            images_batch,
            batch_size=len(images_batch),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for u, e in zip(urls_batch, embs, strict=True):
            url_to_vec[u] = np.asarray(e, dtype=np.float32).ravel()
        images_batch = []
        urls_batch = []

    for u in tqdm(urls, desc="Images (download + CLIP)", unit="url"):
        img = cached_fetch(u, config)
        if img is None:
            continue
        images_batch.append(img)
        urls_batch.append(u)
        if len(images_batch) >= batch_size:
            flush()
    flush()
    return url_to_vec


def collect_unique_urls_from_column(df, img_col: str) -> tuple[dict[Any, list[str]], list[str]]:
    """Per row: parsed URL list; global unique list in first-seen order."""
    row_urls: dict[Any, list[str]] = {}
    seen: set[str] = set()
    ordered_unique: list[str] = []
    for ix in df.index:
        raw = df.loc[ix, img_col]
        urls = parse_image_urls(raw)
        row_urls[ix] = urls
        for u in urls:
            if u not in seen:
                seen.add(u)
                ordered_unique.append(u)
    return row_urls, ordered_unique


def row_embedding_matrices(
    row_urls: dict[Any, list[str]],
    url_to_vec: dict[str, np.ndarray],
) -> dict[Any, np.ndarray]:
    """Stack embeddings per row (shape ki x d). Skip rows with no successful URL."""
    out: dict[Any, np.ndarray] = {}
    for ix, urls in row_urls.items():
        vecs = [url_to_vec[u] for u in urls if u in url_to_vec]
        if not vecs:
            continue
        out[ix] = np.stack(vecs, axis=0)
    return out


def find_similar_groups_multi(
    row_embs: dict[Any, np.ndarray],
    config: AnalyzerConfig,
) -> dict[Any, int]:
    """
    Group rows where max pairwise cosine similarity between any two images
    (across the two rows) exceeds the threshold.
    """
    indices = [k for k, v in row_embs.items() if v.shape[0] >= 1]
    if len(indices) < 2:
        return {}

    normed: dict[Any, np.ndarray] = {
        i: normalize(row_embs[i], norm="l2", axis=1) for i in indices
    }

    parent: dict[Any, Any] = {i: i for i in indices}

    def find(a: Any) -> Any:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: Any, b: Any) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n = len(indices)
    for ia in range(n):
        for jb in range(ia + 1, n):
            i, j = indices[ia], indices[jb]
            sim = float(np.max(normed[i] @ normed[j].T))
            if sim >= config.image_similarity_threshold:
                union(i, j)

    clusters: dict[Any, list[Any]] = defaultdict(list)
    for i in indices:
        clusters[find(i)].append(i)

    row_to_group: dict[Any, int] = {}
    next_gid = 0
    for _root, members in clusters.items():
        if len(members) < 2:
            continue
        for m in members:
            row_to_group[m] = next_gid
        next_gid += 1

    return row_to_group


def run_image_similarity_for_dataframe(df, img_col: str, config: AnalyzerConfig) -> tuple[dict[Any, int], set[Any]]:
    """
    Returns (row -> group_id, set of row labels with at least one embedded image).
    """
    row_urls, unique_urls = collect_unique_urls_from_column(df, img_col)
    url_to_vec = _encode_url_batch(unique_urls, config)
    row_embs = row_embedding_matrices(row_urls, url_to_vec)
    ok_rows = set(row_embs.keys())
    groups = find_similar_groups_multi(row_embs, config)
    return groups, ok_rows
