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
    assert "similar_image_group_id" in out.columns
