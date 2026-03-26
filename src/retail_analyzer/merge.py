from __future__ import annotations

import io
from typing import Any

import pandas as pd
from openpyxl import load_workbook

from retail_analyzer.excel_io import guess_image_column, parse_image_urls

ANALYSIS_COLUMNS = frozenset(
    {
        "duplicate_group_id",
        "duplicate_flag",
        "similar_image_group_id",
        "image_embedding_ok",
        "similarity_score",
        "similarity_comment",
        "anomaly_flags",
        "anomaly_score",
        "anomaly_flag",
        "anomaly_reason",
        "anomaly_type",
        "reason",
    }
)


def merge_visual_duplicate_rows(
    df: pd.DataFrame,
    image_column: str,
    group_column: str | None = None,
    *,
    drop_analysis_columns: bool = True,
    drop_group_id_column: bool = True,
) -> pd.DataFrame:
    """
    Collapse rows that share the same non-negative group_column into one row per group.
    - image_column: unique image refs joined with ", "
    - other string-like columns: unique non-empty values joined with " | " when they differ
    Rows with group_column < 0 are kept unchanged.
    """
    if group_column is None:
        group_column = (
            "duplicate_group_id" if "duplicate_group_id" in df.columns else "similar_image_group_id"
        )
    if group_column not in df.columns:
        raise ValueError(f"Missing column: {group_column}")
    if image_column not in df.columns:
        raise ValueError(f"Missing column: {image_column}")

    dup = df[df[group_column] >= 0].copy()
    single = df[df[group_column] < 0].copy()

    merged_frames: list[pd.DataFrame] = []
    for _gid, g in dup.groupby(group_column, sort=True):
        g = g.sort_index()
        first = g.iloc[0]
        row: dict[str, Any] = {}
        for col in df.columns:
            if col == group_column:
                row[col] = -1
                continue
            if col == image_column:
                refs: list[str] = []
                for _, r in g.iterrows():
                    refs.extend(parse_image_urls(r[image_column]))
                row[col] = ", ".join(list(dict.fromkeys(refs)))
                continue
            if col in ANALYSIS_COLUMNS or str(col).startswith("_detected"):
                row[col] = first[col]
                continue
            vals: list[str] = []
            for _, r in g.iterrows():
                v = r[col]
                if pd.isna(v):
                    continue
                s = str(v).strip()
                if s and s.lower() != "nan":
                    vals.append(s)
            uniq = list(dict.fromkeys(vals))
            if not uniq:
                row[col] = first[col]
            elif len(uniq) == 1:
                row[col] = uniq[0]
            else:
                row[col] = " | ".join(uniq)

        merged_frames.append(pd.DataFrame([row]))

    parts: list[pd.DataFrame] = [single, *merged_frames]
    out = pd.concat(parts, ignore_index=True)

    if drop_analysis_columns:
        drop = [
            c
            for c in out.columns
            if c in ANALYSIS_COLUMNS or str(c).startswith("_detected")
        ]
        out = out.drop(columns=drop, errors="ignore")
    if drop_group_id_column and group_column in out.columns:
        out = out.drop(columns=[group_column])
    return out


def dataframe_to_excel_bytes(df: pd.DataFrame, image_column: str | None = None) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    wb = load_workbook(buf)
    ws = wb.active
    if ws is None:
        out = io.BytesIO()
        wb.save(out)
        return out.getvalue()

    img_col = image_column
    if img_col is None and "_detected_image_column" in df.columns:
        raw = df["_detected_image_column"].iloc[0]
        if isinstance(raw, str) and raw.strip():
            img_col = raw.strip()
    if img_col is None:
        img_col = guess_image_column(list(df.columns))

    if img_col and img_col in df.columns:
        from retail_analyzer.excel_preview import enrich_workbook_with_image_previews

        enrich_workbook_with_image_previews(wb, ws, df, img_col)

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()
