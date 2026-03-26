from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from retail_analyzer.duplicate_detection import run_duplicate_detection
from retail_analyzer.config import AnalyzerConfig
from retail_analyzer.excel_io import (
    guess_category_column,
    guess_description_column,
    guess_image_column,
    guess_price_column,
    load_excel,
    save_excel,
)
from retail_analyzer.excel_style import apply_highlights
from retail_analyzer.retail_anomaly_detection import detect_anomalies

logger = logging.getLogger(__name__)


def analyze_dataframe(
    df: pd.DataFrame,
    *,
    image_column: str | None = None,
    price_column: str | None = None,
    description_column: str | None = None,
    context_spec_columns: list[str] | None = None,
    skip_images: bool = False,
    skip_anomalies: bool = False,
    config: AnalyzerConfig | None = None,
) -> pd.DataFrame:
    """
    Append duplicate and/or anomaly columns.

    ``skip_images=False`` runs duplicate detection (same image URL + embedding similarity).
    ``skip_images=True`` but ``skip_anomalies=False`` still runs URL clustering for
    image–description mismatch anomalies.

    ``skip_anomalies=True`` skips anomaly detection.
    """
    cfg = config or AnalyzerConfig()
    cols = [str(c) for c in df.columns]

    img_col = image_column or guess_image_column(cols)
    price_col = price_column or guess_price_column(cols)
    desc_col = description_column or guess_description_column(cols)
    cat_col = guess_category_column(cols)

    if img_col is None:
        logger.warning("No image column detected; skipping URL-based duplicate checks.")
        skip_images = True
    elif img_col not in df.columns:
        raise ValueError(f"Image column not found: {img_col}")

    out = df.copy()
    idx = df.index
    mismatch: set = set()

    # Defaults (spec columns)
    out["duplicate_group_id"] = pd.Series([-1] * len(idx), index=idx, dtype=int)
    out["duplicate_flag"] = pd.Series([False] * len(idx), index=idx, dtype=bool)
    out["similar_image_group_id"] = pd.Series([-1] * len(idx), index=idx, dtype=int)
    out["image_embedding_ok"] = False
    out["similarity_score"] = np.nan
    out["similarity_comment"] = ""

    run_dup = (not skip_images) or (not skip_anomalies)
    assign_dup_groups = not skip_images

    if run_dup and img_col:
        dup_id, dup_flag, sim_score, sim_comment, mismatch = run_duplicate_detection(
            df,
            img_col,
            desc_col,
            cat_col,
            cfg,
            assign_duplicate_groups=assign_dup_groups,
        )
        out["duplicate_group_id"] = dup_id
        out["duplicate_flag"] = dup_flag
        out["similar_image_group_id"] = dup_id
        out["similarity_score"] = sim_score
        out["similarity_comment"] = sim_comment.astype(str)

    if skip_anomalies:
        out["anomaly_flag"] = pd.Series([False] * len(idx), index=idx, dtype=bool)
        out["anomaly_reason"] = pd.Series([""] * len(idx), index=idx, dtype=object)
        out["anomaly_flags"] = pd.Series([""] * len(idx), index=idx, dtype=object)
        out["anomaly_score"] = pd.Series([0.0] * len(idx), index=idx, dtype=float)
        out["anomaly_type"] = pd.Series([""] * len(idx), index=idx, dtype=object)
        out["reason"] = pd.Series([""] * len(idx), index=idx, dtype=object)
    else:
        (
            a_flag,
            a_reason,
            a_codes,
            a_score,
            a_type,
            reason_col,
        ) = detect_anomalies(
            df,
            img_col,
            price_col,
            desc_col,
            cfg,
            category_col=cat_col,
            context_spec_columns=context_spec_columns,
            image_description_mismatch_rows=mismatch,
        )
        out["anomaly_flag"] = a_flag
        out["anomaly_reason"] = a_reason
        out["anomaly_flags"] = a_codes
        out["anomaly_score"] = a_score.round(3)
        out["anomaly_type"] = a_type
        out["reason"] = reason_col

    out["_detected_image_column"] = img_col or ""
    out["_detected_price_column"] = price_col or ""
    out["_detected_description_column"] = desc_col or ""
    out["_detected_category_column"] = cat_col or ""

    return out


def run_analysis(
    input_path: Path,
    output_path: Path | None = None,
    *,
    sheet: str | int | None = 0,
    image_column: str | None = None,
    price_column: str | None = None,
    description_column: str | None = None,
    context_spec_columns: list[str] | None = None,
    skip_images: bool = False,
    skip_anomalies: bool = False,
    config: AnalyzerConfig | None = None,
) -> pd.DataFrame:
    """
    Load Excel, analyze, optionally write highlighted workbook.
    """
    df = load_excel(input_path, sheet=sheet)
    out = analyze_dataframe(
        df,
        image_column=image_column,
        price_column=price_column,
        description_column=description_column,
        context_spec_columns=context_spec_columns,
        skip_images=skip_images,
        skip_anomalies=skip_anomalies,
        config=config,
    )
    if output_path is not None:
        save_excel(out, output_path)
        apply_highlights(output_path, out)
    return out
