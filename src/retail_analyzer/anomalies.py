from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from retail_analyzer.config import AnalyzerConfig
from retail_analyzer.excel_io import parse_image_urls


def _series_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma is None or sigma == 0 or np.isnan(sigma):
        return pd.Series(0.0, index=s.index)
    z = (s - mu) / sigma
    return z.abs()


def detect_anomalies(
    df: pd.DataFrame,
    image_col: str | None,
    price_col: str | None,
    desc_col: str | None,
    config: AnalyzerConfig,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns (flags: semicolon-separated strings, score: float 0..1 rough severity).
    """
    flags: list[str] = []
    scores: list[float] = []

    n = len(df)
    idx = df.index

    for ix in idx:
        row = df.loc[ix]
        parts: list[str] = []
        sev = 0.0

        if image_col and image_col in df.columns:
            v = row[image_col]
            parsed = parse_image_urls(v)
            if not parsed:
                parts.append("missing_image_url")
                sev += 0.35

        if price_col and price_col in df.columns:
            p = pd.to_numeric(row[price_col], errors="coerce")
            if pd.isna(p):
                parts.append("missing_price")
                sev += 0.3
            elif p <= 0:
                parts.append("non_positive_price")
                sev += 0.35

        if desc_col and desc_col in df.columns:
            t = row[desc_col]
            text = "" if pd.isna(t) else str(t).strip()
            if len(text) < config.min_description_length:
                parts.append("short_or_missing_description")
                sev += 0.25

        flags.append(";".join(parts) if parts else "")
        scores.append(min(1.0, sev))

    flag_series = pd.Series(flags, index=idx)
    score_series = pd.Series(scores, index=idx)

    # Global: same image URL reused on another row (supports comma-separated cells)
    if image_col and image_col in df.columns:
        flat: list[str] = []
        for ix in idx:
            flat.extend(parse_image_urls(df.loc[ix, image_col]))
        counts = Counter(flat)
        dup_urls = {u for u, c in counts.items() if c > 1}
        for ix in idx:
            row_urls = parse_image_urls(df.loc[ix, image_col])
            if any(u in dup_urls for u in row_urls):
                prev = flag_series.loc[ix]
                flag_series.loc[ix] = (
                    str(prev) + ";duplicate_exact_image_url" if prev else "duplicate_exact_image_url"
                )
                score_series.loc[ix] = min(1.0, float(score_series.loc[ix]) + 0.3)

    # Price outliers (z-score) when enough valid prices
    if price_col and price_col in df.columns:
        s = pd.to_numeric(df[price_col], errors="coerce")
        valid = s.notna() & (s > 0)
        if valid.sum() >= 8:
            z = _series_zscore(s)
            mask = z > config.price_zscore_threshold
            hit = mask & valid
            for ix in df.index[hit]:
                prev = flag_series.loc[ix]
                flag_series.loc[ix] = (
                    str(prev) + ";price_outlier_zscore" if prev else "price_outlier_zscore"
                )
                score_series.loc[ix] = min(1.0, float(score_series.loc[ix]) + 0.25)

    # Multivariate numeric anomaly (IsolationForest) on numeric columns
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2 and len(num) >= 10:
        X = num.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if len(X) >= 10:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X.values)
            iso = IsolationForest(
                random_state=42,
                contamination="auto",
                n_estimators=200,
            )
            pred = iso.fit_predict(Xs)
            # -1 = anomaly in sklearn
            anom_idx = set(X.index[pred == -1])
            for ix in anom_idx:
                prev = flag_series.loc[ix]
                flag_series.loc[ix] = (
                    str(prev) + ";numeric_row_outlier" if prev else "numeric_row_outlier"
                )
                score_series.loc[ix] = min(1.0, float(score_series.loc[ix]) + 0.2)

    return flag_series, score_series
