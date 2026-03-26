"""
Retail anomaly detection aligned with product-catalog spec.

anomaly_type values: image_description_mismatch, price_outlier, missing_data, invalid_format
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from retail_analyzer.config import AnalyzerConfig
from retail_analyzer.excel_io import parse_image_urls

logger = logging.getLogger(__name__)

TYPE_MESSAGES: dict[str, str] = {
    "image_description_mismatch": "Same image URL as another row but descriptions are very different.",
    "price_outlier": "Price is unusually high or low compared with other products in the same category (beyond 3 standard deviations).",
    "missing_data": "A required field (price, description, category, or image URL) is empty.",
    "invalid_format": "Price is not numeric or the image URL format is not valid.",
}

ANOMALY_HELP_BULLETS: tuple[str, ...] = tuple(TYPE_MESSAGES.values())


def _non_empty(val: object) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    return bool(s) and s.lower() != "nan"


def _looks_like_url_attempt(s: str) -> bool:
    s = s.strip().lower()
    return "http" in s or "www." in s or s.startswith("ftp")


def _invalid_price_when_present(val: object) -> bool:
    if not _non_empty(val):
        return False
    return pd.isna(pd.to_numeric(val, errors="coerce"))


def _invalid_image_url_when_present(row: pd.Series, image_col: str) -> bool:
    if image_col not in row.index:
        return False
    v = row[image_col]
    if not _non_empty(v):
        return False
    if not _looks_like_url_attempt(str(v).strip()):
        return False
    return len(parse_image_urls(v)) == 0


def _price_outliers_mean_std(
    df: pd.DataFrame,
    price_col: str,
    category_col: str,
    config: AnalyzerConfig,
) -> set[Any]:
    """Flag prices outside mean ± k*std within each category (raw prices, positive only)."""
    out: set[Any] = set()
    k = config.price_outlier_std_multiplier
    min_n = config.min_category_group_size
    for _cat, sub in df.groupby(category_col, dropna=False):
        if len(sub) < min_n:
            continue
        prices = pd.to_numeric(sub[price_col], errors="coerce")
        valid = prices.notna() & (prices > 0)
        if int(valid.sum()) < min_n:
            continue
        pv = prices[valid]
        mu = float(pv.mean())
        sigma = float(pv.std(ddof=0))
        if sigma < 1e-12:
            continue
        for ix in pv.index:
            p = float(prices.loc[ix])
            if abs(p - mu) > k * sigma:
                out.add(ix)
    return out


def detect_anomalies(
    df: pd.DataFrame,
    image_col: str | None,
    price_col: str | None,
    desc_col: str | None,
    config: AnalyzerConfig,
    *,
    category_col: str | None = None,
    context_spec_columns: list[str] | None = None,
    image_description_mismatch_rows: set[Any] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns:
      anomaly_flag (bool),
      anomaly_reason (human text),
      anomaly_flags (semicolon type codes),
      anomaly_score,
      anomaly_type (semicolon-separated types),
      reason (same as anomaly_reason for spec output)
    """
    del context_spec_columns
    idx = df.index
    types_map: dict[Any, list[str]] = {ix: [] for ix in idx}
    mismatch_rows = image_description_mismatch_rows or set()

    def add_type(ix: Any, t: str) -> None:
        if t not in types_map[ix]:
            types_map[ix].append(t)

    for ix in mismatch_rows:
        if ix in idx:
            add_type(ix, "image_description_mismatch")

    # Missing critical fields
    for ix in idx:
        row = df.loc[ix]
        if price_col and price_col in df.columns:
            if not _non_empty(row.get(price_col)):
                add_type(ix, "missing_data")
        if desc_col and desc_col in df.columns:
            if not _non_empty(row.get(desc_col)):
                add_type(ix, "missing_data")
        if category_col and category_col in df.columns:
            if not _non_empty(row.get(category_col)):
                add_type(ix, "missing_data")
        if image_col and image_col in df.columns:
            if not _non_empty(row.get(image_col)):
                add_type(ix, "missing_data")

    # Invalid format (when cell is non-empty)
    if price_col and price_col in df.columns:
        for ix in idx:
            v = df.loc[ix, price_col]
            if _invalid_price_when_present(v):
                add_type(ix, "invalid_format")
            else:
                p = pd.to_numeric(v, errors="coerce")
                if _non_empty(v) and not pd.isna(p) and p <= 0:
                    add_type(ix, "invalid_format")

    if image_col and image_col in df.columns:
        for ix in idx:
            if _invalid_image_url_when_present(df.loc[ix], image_col):
                add_type(ix, "invalid_format")

    # Price outliers (same category, mean ± 3 std)
    if (
        category_col
        and category_col in df.columns
        and price_col
        and price_col in df.columns
    ):
        for ix in _price_outliers_mean_std(df, price_col, category_col, config):
            add_type(ix, "price_outlier")

    flag_list: list[str] = []
    reason_list: list[str] = []
    code_list: list[str] = []
    score_list: list[float] = []
    atype_list: list[str] = []

    for ix in idx:
        types = types_map[ix]
        atype_str = ";".join(types) if types else ""
        parts = [TYPE_MESSAGES[t] for t in types if t in TYPE_MESSAGES]
        reason_str = " ".join(dict.fromkeys(parts)) if parts else ""
        sev = min(1.0, 0.15 * len(types) + 0.05 * max(0, len(types) - 1))
        flag_list.append(bool(types))
        reason_list.append(reason_str)
        code_list.append(atype_str)
        score_list.append(sev)
        atype_list.append(atype_str)

    return (
        pd.Series(flag_list, index=idx),
        pd.Series(reason_list, index=idx),
        pd.Series(code_list, index=idx),
        pd.Series(score_list, index=idx),
        pd.Series(atype_list, index=idx),
        pd.Series(reason_list, index=idx),
    )
