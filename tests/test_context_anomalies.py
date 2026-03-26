import pandas as pd

from retail_analyzer.config import AnalyzerConfig
from retail_analyzer.context_anomalies import (
    detect_context_row,
    extract_measurements,
    extract_currency_amounts,
)


def test_extract_inch():
    m = extract_measurements('Smart TV 40 inch LED, wall mount')
    vals = [v for v, u in m if u == "length_in"]
    assert 40.0 in vals


def test_typo_40_vs_4_inch():
    cfg = AnalyzerConfig()
    row = pd.Series(
        {
            "title": "Premium 40 inch 4K UHD television",
            "size_note": "4 inch screen (typo)",
        }
    )
    flags = detect_context_row(
        row,
        ["title", "size_note"],
        desc_col="title",
        image_col=None,
        price_col=None,
        spec_columns=["size_note"],
        config=cfg,
    )
    assert any("likely_typo" in f or "mismatch" in f for f in flags)


def test_price_text_mismatch():
    cfg = AnalyzerConfig()
    row = pd.Series(
        {
            "desc": "Offer price ₹50,000 only today",
            "price": 5000.0,
        }
    )
    flags = detect_context_row(
        row,
        ["desc", "price"],
        desc_col="desc",
        image_col=None,
        price_col="price",
        spec_columns=None,
        config=cfg,
    )
    assert any("price_in_text" in f for f in flags)


def test_extract_currency():
    assert 50000.0 in extract_currency_amounts("Priced at ₹50,000 inclusive")
