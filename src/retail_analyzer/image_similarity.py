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


def compute_duplicate_row_scores_and_comments(
    row_embs: dict[Any, np.ndarray],
    groups: dict[Any, int],
    threshold: float,
    row_num: dict[Any, int],
) -> tuple[dict[Any, float], dict[Any, str]]:
    """
    For each row in a duplicate group, compute the strongest cosine similarity to another
    row in the same group and a short human-readable summary.
    """
    if not groups:
        return {}, {}

    clusters: dict[int, list[Any]] = defaultdict(list)
    for ix, gid in groups.items():
        clusters[gid].append(ix)

    scores: dict[Any, float] = {}
    comments: dict[Any, str] = {}
    thr = float(threshold)

    for ix, gid in groups.items():
        members = clusters[gid]
        normed_i = normalize(row_embs[ix], norm="l2", axis=1)
        best_sim = 0.0
        best_partner: Any = None
        for j in members:
            if j == ix:
                continue
            normed_j = normalize(row_embs[j], norm="l2", axis=1)
            sim = float(np.max(normed_i @ normed_j.T))
            if sim > best_sim:
                best_sim = sim
                best_partner = j
        if best_partner is None:
            continue
        scores[ix] = best_sim
        rn_partner = row_num.get(best_partner, best_partner)
        n_in_group = len(members)
        pct_match = best_sim * 100.0
        pct_min = thr * 100.0
        others = n_in_group - 1
        others_phrase = (
            f"{others} other listing(s) in this group also look like the same product."
            if others != 1
            else "One other listing in this group also looks like the same product."
        )
        comments[ix] = (
            f"The product image here looks like the same item as row {rn_partner} "
            f"(roughly {pct_match:.0f}% match). "
            f"{others_phrase} "
            f"You set the tool to group rows when images are at least {pct_min:.0f}% similar."
        )
    return scores, comments


def run_image_similarity_for_dataframe(
    df, img_col: str, config: AnalyzerConfig
) -> tuple[dict[Any, int], set[Any], dict[Any, float], dict[Any, str]]:
    """
    Returns (row -> group_id, rows with at least one embedded image,
    per-row max similarity within group, per-row summary comment).
    """
    row_urls, unique_urls = collect_unique_urls_from_column(df, img_col)
    url_to_vec = _encode_url_batch(unique_urls, config)
    row_embs = row_embedding_matrices(row_urls, url_to_vec)
    ok_rows = set(row_embs.keys())
    groups = find_similar_groups_multi(row_embs, config)
    row_num = {ix: i + 1 for i, ix in enumerate(df.index)}
    scores: dict[Any, float] = {}
    comments: dict[Any, str] = {}
    if groups:
        scores, comments = compute_duplicate_row_scores_and_comments(
            row_embs,
            groups,
            config.image_similarity_threshold,
            row_num,
        )
    return groups, ok_rows, scores, comments
