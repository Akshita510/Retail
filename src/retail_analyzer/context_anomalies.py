"""
Cross-field context checks: compare product description to other text/numeric fields
without hardcoding product categories. Focus: measurements, obvious typos (e.g. 4 vs 40 inch),
and optional price-in-text vs price column.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from retail_analyzer.config import AnalyzerConfig

# (value, unit_family) — unit_family used only for apples-to-apples comparison
_MEASURE_RES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|in\.?)(?!\w)", re.I), "length_in"),
    (re.compile(r'(\d+(?:\.\d+)?)\s*(?:["\u2033\u201d])'), "length_in"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*cm\b", re.I), "length_cm"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", re.I), "length_mm"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*m\b", re.I), "length_m"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*ft\b", re.I), "length_ft"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*kg\b", re.I), "mass_kg"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*g\b(?![a-z])", re.I), "mass_g"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*lb\b", re.I), "mass_lb"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*oz\b", re.I), "mass_oz"),
]

# Price-like amounts in marketing copy (Indian + Western)
_CURRENCY_RES: list[re.Pattern[str]] = [
    re.compile(r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)", re.I),
    re.compile(r"\$\s*([\d,]+(?:\.\d+)?)"),
    re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:₹|USD|EUR)", re.I),
]


def _parse_num(s: str) -> float | None:
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_measurements(text: str) -> list[tuple[float, str]]:
    """Return (value, unit_family) for each measurement found."""
    if not text or not str(text).strip():
        return []
    t = str(text)
    out: list[tuple[float, str]] = []
    for pat, family in _MEASURE_RES:
        for m in pat.finditer(t):
            v = _parse_num(m.group(1))
            if v is not None and v > 0:
                out.append((v, family))
    return out


def extract_currency_amounts(text: str) -> list[float]:
    """Rough extraction of monetary amounts from a blob of text."""
    if not text:
        return []
    t = str(text)
    vals: list[float] = []
    for pat in _CURRENCY_RES:
        for m in pat.finditer(t):
            v = _parse_num(m.group(1))
            if v is not None and v > 0:
                vals.append(v)
    return vals


def _is_likely_missing_digit_typo(a: float, b: float, lo: float, hi: float) -> bool:
    """e.g. 4 vs 40 inch: ~10× ratio and smaller integer is a prefix of the larger."""
    if a <= 0 or b <= 0:
        return False
    big, small = (a, b) if a >= b else (b, a)
    r = big / small
    if not (lo <= r <= hi):
        return False
    ib, is_ = int(round(big)), int(round(small))
    if is_ <= 0 or ib <= 0:
        return False
    return str(is_) == str(ib)[: len(str(is_))]


def _cross_field_flags(
    desc_meas: list[tuple[float, str]],
    other_meas: list[tuple[float, str]],
    config: AnalyzerConfig,
) -> list[str]:
    flags: list[str] = []
    lo, hi = config.context_typo_ratio_lo, config.context_typo_ratio_hi
    rel_tol = config.context_dimension_rel_tolerance

    for vd, fd in desc_meas:
        for vo, fo in other_meas:
            if fd != fo:
                continue
            if abs(vd - vo) < 1e-6:
                continue
            if _is_likely_missing_digit_typo(vd, vo, lo, hi):
                flags.append("likely_typo_dimension_vs_description")
                continue
            mx = max(vd, vo)
            if mx > 0 and abs(vd - vo) / mx > rel_tol:
                flags.append("cross_field_dimension_mismatch")
    return list(dict.fromkeys(flags))


def _price_text_vs_column(
    desc_amounts: list[float],
    price_val: Any,
    config: AnalyzerConfig,
) -> list[str]:
    flags: list[str] = []
    p = pd.to_numeric(price_val, errors="coerce")
    if pd.isna(p) or p <= 0 or not desc_amounts:
        return flags
    # If description mentions a price and it disagrees strongly with the price column
    for d in desc_amounts:
        if d <= 0:
            continue
        ratio = max(d, float(p)) / min(d, float(p)) if min(d, float(p)) > 0 else 999
        if ratio >= config.context_price_text_mismatch_ratio:
            flags.append("price_in_text_vs_column_mismatch")
            break
    return flags


_EXCLUDE_SPEC_NOISE = re.compile(
    r"(ingredient|nutrition|allergen|validation|comment|ops|flag|foreign|language|nutrient|serving)",
    re.I,
)
_INCLUDE_SPEC = re.compile(
    r"(^|_)(spec|size|dim|screen|weight|package|variant|model|sku|type|title|name|subtitle|detail|feature|attr|height|width|length)",
    re.I,
)


def guess_spec_text_columns(
    columns: list[str],
    *,
    desc_col: str | None,
    image_col: str | None,
    price_col: str | None,
) -> list[str]:
    """
    Prefer short, structured spec columns — avoid long ingredient/nutrition/validation blobs
    that cause false cross-field measurement matches.
    """
    skip = {c for c in (desc_col, image_col, price_col) if c}
    out: list[str] = []
    for c in columns:
        if c in skip or str(c).startswith("_"):
            continue
        cs = str(c)
        if _EXCLUDE_SPEC_NOISE.search(cs):
            continue
        if _INCLUDE_SPEC.search(cs):
            out.append(c)
    if out:
        return out
    # Fallback: title/name/type columns only (still exclude noise by name)
    for c in columns:
        if c in skip or str(c).startswith("_"):
            continue
        if _EXCLUDE_SPEC_NOISE.search(str(c)):
            continue
        if re.search(r"(product_name|product_type|title|name|subtitle|model)", str(c), re.I):
            out.append(c)
    return out


def detect_context_row(
    row: pd.Series,
    df_columns: list[str],
    desc_col: str | None,
    image_col: str | None,
    price_col: str | None,
    spec_columns: list[str] | None,
    config: AnalyzerConfig,
) -> list[str]:
    if not config.enable_context_anomalies or not desc_col or desc_col not in row.index:
        return []

    desc_text = "" if pd.isna(row[desc_col]) else str(row[desc_col])
    if len(desc_text.strip()) < 3:
        return []

    desc_meas = extract_measurements(desc_text)
    desc_money = extract_currency_amounts(desc_text)

    cols = spec_columns
    if not cols:
        cols = guess_spec_text_columns(df_columns, desc_col=desc_col, image_col=image_col, price_col=price_col)

    flags: list[str] = []

    # Merge measurements from spec-like columns (skip huge cells — often bilingual ingredients)
    max_cell = config.context_max_spec_cell_chars
    other_meas: list[tuple[float, str]] = []
    for c in cols:
        if c == desc_col or c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if len(s) < 2:
            continue
        if len(s) > max_cell and _EXCLUDE_SPEC_NOISE.search(str(c)):
            continue
        if len(s) > max_cell * 2:
            continue
        other_meas.extend(extract_measurements(s))

    # Numeric cells that are plain numbers (might be "4" in a size-only cell) — only if column name hints size
    size_hint = re.compile(r"(size|dim|screen|inch|cm|mm|length|width)", re.I)
    for c in df_columns:
        if c == desc_col or not size_hint.search(str(c)):
            continue
        if c not in row.index:
            continue
        raw = row[c]
        num = pd.to_numeric(raw, errors="coerce")
        if pd.isna(num) or float(num) <= 0:
            continue
        # Treat bare number in a size column as inches if description uses inches heavily
        if any(u == "length_in" for _, u in desc_meas):
            other_meas.append((float(num), "length_in"))

    flags.extend(_cross_field_flags(desc_meas, other_meas, config))

    if price_col and price_col in row.index and desc_money:
        flags.extend(_price_text_vs_column(desc_money, row[price_col], config))

    return list(dict.fromkeys(flags))


def merge_context_anomalies(
    df: pd.DataFrame,
    flag_series: pd.Series,
    score_series: pd.Series,
    desc_col: str | None,
    image_col: str | None,
    price_col: str | None,
    spec_columns: list[str] | None,
    config: AnalyzerConfig,
) -> tuple[pd.Series, pd.Series]:
    """Append context flags and bump scores."""
    cols = [str(c) for c in df.columns]
    idx = df.index
    for ix in idx:
        parts = detect_context_row(
            df.loc[ix],
            cols,
            desc_col,
            image_col,
            price_col,
            spec_columns,
            config,
        )
        if not parts:
            continue
        prev = str(flag_series.loc[ix])
        for p in parts:
            flag_series.loc[ix] = f"{prev};{p}" if prev else p
        flag_series.loc[ix] = flag_series.loc[ix].strip(";")
        score_series.loc[ix] = min(
            1.0,
            float(score_series.loc[ix]) + 0.25 * len(parts),
        )
    return flag_series, score_series
