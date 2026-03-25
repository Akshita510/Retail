from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from retail_analyzer.anomalies import detect_anomalies
from retail_analyzer.config import AnalyzerConfig
from retail_analyzer.excel_io import (
    guess_description_column,
    guess_image_column,
    guess_price_column,
    load_excel,
    save_excel,
)
from retail_analyzer.excel_style import apply_highlights
from retail_analyzer.image_similarity import run_image_similarity_for_dataframe

logger = logging.getLogger(__name__)


def analyze_dataframe(
    df: pd.DataFrame,
    *,
    image_column: str | None = None,
    price_column: str | None = None,
    description_column: str | None = None,
    skip_images: bool = False,
    config: AnalyzerConfig | None = None,
) -> pd.DataFrame:
    """
    Append similarity + anomaly columns. Does not read or write files.
    """
    cfg = config or AnalyzerConfig()
    cols = [str(c) for c in df.columns]

    img_col = image_column or guess_image_column(cols)
    price_col = price_column or guess_price_column(cols)
    desc_col = description_column or guess_description_column(cols)

    if img_col is None:
        logger.warning("No image column detected; set image_column. Skipping CLIP similarity.")
        skip_images = True
    elif img_col not in df.columns:
        raise ValueError(f"Image column not found: {img_col}")

    out = df.copy()
    out["similar_image_group_id"] = -1
    out["image_embedding_ok"] = False

    if not skip_images and img_col:
        groups, ok_rows = run_image_similarity_for_dataframe(df, img_col, cfg)
        for ix in ok_rows:
            out.loc[ix, "image_embedding_ok"] = True
        for ix, gid in groups.items():
            out.loc[ix, "similar_image_group_id"] = gid

    flags, scores = detect_anomalies(df, img_col, price_col, desc_col, cfg)
    out["anomaly_flags"] = flags
    out["anomaly_score"] = scores.round(3)

    out["_detected_image_column"] = img_col or ""
    out["_detected_price_column"] = price_col or ""
    out["_detected_description_column"] = desc_col or ""

    return out


def run_analysis(
    input_path: Path,
    output_path: Path | None = None,
    *,
    sheet: str | int | None = 0,
    image_column: str | None = None,
    price_column: str | None = None,
    description_column: str | None = None,
    skip_images: bool = False,
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
        skip_images=skip_images,
        config=config,
    )
    if output_path is not None:
        save_excel(out, output_path)
        apply_highlights(output_path, out)
    return out
