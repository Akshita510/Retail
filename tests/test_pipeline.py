import pandas as pd

from retail_analyzer.pipeline import analyze_dataframe


def test_analyze_dataframe_skip_images():
    df = pd.DataFrame(
        {
            "product_name": ["x"],
            "image_urls": ["https://example.com/a.jpg"],
            "price": [1.0],
        }
    )
    out = analyze_dataframe(
        df,
        image_column="image_urls",
        price_column="price",
        description_column="product_name",
        skip_images=True,
    )
    assert "anomaly_flags" in out.columns
    assert "duplicate_group_id" in out.columns
    assert "similar_image_group_id" in out.columns
    assert "similarity_score" in out.columns
    assert out["similarity_score"].isna().all()
    assert "anomaly_flag" in out.columns
    assert "anomaly_type" in out.columns
    assert "reason" in out.columns


def test_analyze_dataframe_skip_anomalies():
    df = pd.DataFrame(
        {
            "product_name": ["x"],
            "image_urls": ["https://example.com/a.jpg"],
            "price": [1.0],
        }
    )
    out = analyze_dataframe(
        df,
        image_column="image_urls",
        price_column="price",
        description_column="product_name",
        skip_images=True,
        skip_anomalies=True,
    )
    assert (out["anomaly_flags"] == "").all()
    assert (out["anomaly_score"] == 0.0).all()
    assert not out["anomaly_flag"].any()
    assert (out["anomaly_reason"] == "").all()
