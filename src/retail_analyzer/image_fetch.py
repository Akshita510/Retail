from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

import requests
from PIL import Image

from retail_analyzer.config import AnalyzerConfig
from retail_analyzer.excel_io import resolve_image_path

logger = logging.getLogger(__name__)


def _url_cache_key(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def fetch_image_pil(url: str, config: AnalyzerConfig) -> Image.Image | None:
    """Load image from local path or HTTP URL; return RGB PIL image, or None on failure."""
    if not isinstance(url, str) or not url.strip():
        return None
    u = url.strip()
    local = resolve_image_path(u)
    if local is not None:
        try:
            img = Image.open(local)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:  # noqa: BLE001
            logger.debug("local image open failed %s: %s", local, e)
            return None
    try:
        r = requests.get(
            u,
            timeout=config.request_timeout_sec,
            headers={"User-Agent": "RetailExcelAnalyzer/1.0"},
        )
        r.raise_for_status()
        if len(r.content) > config.max_image_bytes:
            logger.warning("Image too large, skipping: %s", u[:80])
            return None
        img = Image.open(io.BytesIO(r.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:  # noqa: BLE001 — many network/PIL errors
        logger.debug("fetch failed %s: %s", u[:80], e)
        return None


def cached_fetch(url: str, config: AnalyzerConfig) -> Image.Image | None:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    key = _url_cache_key(url)
    path = config.cache_dir / f"{key}.jpg"
    if path.is_file():
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception:
            path.unlink(missing_ok=True)
    img = fetch_image_pil(url, config)
    if img is None:
        return None
    try:
        img.save(path, format="JPEG", quality=90)
    except Exception:
        pass
    return img
