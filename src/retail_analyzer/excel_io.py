from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd


def parse_image_urls(value: object) -> list[str]:
    """
    Split a cell that may hold multiple HTTP(S) links (comma/semicolon/newline separated).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    parts = re.split(r"[,;\n]+", s)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) >= 8 and p.lower().startswith("http"):
            out.append(p)
    return out


def load_excel(path: Path, sheet: str | int | None = 0) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_excel(p, sheet_name=sheet, engine="openpyxl")


def load_excel_bytes(data: bytes, sheet: str | int | None = 0) -> pd.DataFrame:
    """Load the first (or selected) sheet from an in-memory .xlsx."""
    buf = io.BytesIO(data)
    return pd.read_excel(buf, sheet_name=sheet, engine="openpyxl")


def save_excel(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, engine="openpyxl")


def guess_image_column(columns: list[str]) -> str | None:
    """Pick a column that likely holds image URLs."""
    patterns = (
        r"image",
        r"img",
        r"photo",
        r"picture",
        r"url",
        r"link",
        r"thumbnail",
    )
    lowered = [(c, str(c).lower()) for c in columns]
    for pat in patterns:
        for orig, low in lowered:
            if re.search(pat, low):
                return str(orig)
    return None


def guess_price_column(columns: list[str]) -> str | None:
    for c in columns:
        low = str(c).lower()
        if any(x in low for x in ("price", "cost", "mrp", "amount", "rate")):
            return str(c)
    return None


def guess_description_column(columns: list[str]) -> str | None:
    for c in columns:
        low = str(c).lower()
        if any(x in low for x in ("desc", "title", "name", "product")):
            return str(c)
    return None


def list_sheet_names_bytes(data: bytes) -> list[str]:
    """Return sheet names from an xlsx in memory."""
    buf = io.BytesIO(data)
    xl = pd.ExcelFile(buf, engine="openpyxl")
    return list(xl.sheet_names)
