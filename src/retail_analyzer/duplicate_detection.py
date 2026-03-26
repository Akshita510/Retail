"""
Duplicate detection by same image URL + description/category similarity (sentence embeddings).

Rules (spec):
- duplicate: same normalized image URL AND description cosine > dup threshold AND category cosine > cat threshold
- same URL, description similarity < anomaly threshold → flagged for image_description_mismatch (handled in anomaly layer)
- uncertain band: optional note in similarity_comment
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from retail_analyzer.config import AnalyzerConfig
from retail_analyzer.embeddings import embed_texts
from retail_analyzer.excel_io import parse_image_urls

logger = logging.getLogger(__name__)


def _primary_url(cell: object) -> str | None:
    urls = parse_image_urls(cell)
    if not urls:
        return None
    return urls[0].strip().lower().rstrip("/")


def _desc_text(row: pd.Series, desc_col: str | None) -> str:
    if not desc_col or desc_col not in row.index:
        return ""
    v = row[desc_col]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()[:4000]


def _cat_text(row: pd.Series, cat_col: str | None) -> str:
    if not cat_col or cat_col not in row.index:
        return ""
    v = row[cat_col]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()[:500]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def run_duplicate_detection(
    df: pd.DataFrame,
    image_col: str,
    desc_col: str | None,
    cat_col: str | None,
    config: AnalyzerConfig,
    *,
    assign_duplicate_groups: bool = True,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    set[Any],
]:
    """
    Returns:
      duplicate_group_id (int, -1 if none)
      duplicate_flag (bool)
      similarity_score (float, nan if not duplicate) — min pairwise desc sim within duplicate component
      similarity_comment (str)
      image_description_mismatch_indices — rows to flag for anomaly_type image_description_mismatch
    """
    idx = df.index
    n = len(df)
    dup_id = pd.Series([-1] * n, index=idx, dtype=int)
    dup_flag = pd.Series([False] * n, index=idx, dtype=bool)
    sim_score = pd.Series([np.nan] * n, index=idx, dtype=float)
    sim_comment = pd.Series([""] * n, index=idx, dtype=object)
    mismatch: set[Any] = set()

    if not image_col or image_col not in df.columns:
        return dup_id, dup_flag, sim_score, sim_comment, mismatch

    if not desc_col or desc_col not in df.columns:
        logger.warning("No description column; cannot compute description similarity for duplicates.")
        return dup_id, dup_flag, sim_score, sim_comment, mismatch

    # Cluster by primary URL
    url_to_indices: dict[str, list[Any]] = defaultdict(list)
    for i, ix in enumerate(idx):
        u = _primary_url(df.loc[ix, image_col])
        if u:
            url_to_indices[u].append(ix)

    next_gid = 0

    for _url, members in url_to_indices.items():
        if len(members) < 2:
            continue
        k = len(members)
        descs = [_desc_text(df.loc[ix], desc_col) for ix in members]
        cats = [_cat_text(df.loc[ix], cat_col) for ix in members]

        # Embeddings for descriptions (empty -> small placeholder to avoid collapse)
        desc_in = [d if d else "(no description)" for d in descs]
        emb_d = embed_texts(desc_in)
        emb_d_n = normalize(emb_d, norm="l2", axis=1)
        sim_desc = emb_d_n @ emb_d_n.T

        # Category similarity: embedding of category text (if no category column, treat as same category)
        if cat_col and cat_col in df.columns:
            cat_in = [c if c else "(no category)" for c in cats]
            emb_c = embed_texts(cat_in)
            emb_c_n = normalize(emb_c, norm="l2", axis=1)
            sim_cat = emb_c_n @ emb_c_n.T
        else:
            sim_cat = np.ones((k, k), dtype=np.float64)

        dup_thr = config.duplicate_desc_cosine_threshold
        cat_thr = config.duplicate_category_cosine_threshold
        anom_thr = config.same_image_desc_anomaly_threshold

        # Pairwise anomalies: same image, very different description
        for i in range(k):
            for j in range(i + 1, k):
                if sim_desc[i, j] < anom_thr:
                    mismatch.add(members[i])
                    mismatch.add(members[j])

        uncertain_parts: list[str] = []
        for i in range(k):
            for j in range(i + 1, k):
                s = float(sim_desc[i, j])
                if anom_thr <= s < dup_thr:
                    uncertain_parts.append(f"pair({i},{j})~{s:.2f}")

        uf = _UnionFind(k)
        for i in range(k):
            for j in range(i + 1, k):
                if sim_desc[i, j] >= dup_thr and sim_cat[i, j] >= cat_thr:
                    uf.union(i, j)

        roots: dict[int, list[int]] = defaultdict(list)
        for i in range(k):
            roots[uf.find(i)].append(i)

        if assign_duplicate_groups:
            for _root, pos in roots.items():
                if len(pos) < 2:
                    continue
                gid = next_gid
                next_gid += 1
                pos_list = pos
                min_sim = 1.0
                for a in range(len(pos_list)):
                    for b in range(a + 1, len(pos_list)):
                        ii, jj = pos_list[a], pos_list[b]
                        min_sim = min(min_sim, float(sim_desc[ii, jj]))
                for p in pos_list:
                    ix = members[p]
                    dup_id.loc[ix] = gid
                    dup_flag.loc[ix] = True
                    sim_score.loc[ix] = round(min_sim, 4)
                    sim_comment.loc[ix] = "same_image_similar_description (category aligned)."
            if uncertain_parts and k >= 2 and assign_duplicate_groups:
                note = " Uncertain: some pairs in this image-URL cluster have similarity between 0.4 and 0.8 (manual review)."
                for ix in members:
                    if dup_flag.loc[ix]:
                        sim_comment.loc[ix] = str(sim_comment.loc[ix]) + note

    if not assign_duplicate_groups:
        dup_id = pd.Series([-1] * n, index=idx, dtype=int)
        dup_flag = pd.Series([False] * n, index=idx, dtype=bool)
        sim_score = pd.Series([np.nan] * n, index=idx, dtype=float)
        sim_comment = pd.Series([""] * n, index=idx, dtype=object)

    return dup_id, dup_flag, sim_score, sim_comment, mismatch
