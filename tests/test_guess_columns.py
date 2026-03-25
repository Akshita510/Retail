from retail_analyzer.excel_io import (
    guess_description_column,
    guess_image_column,
    guess_price_column,
    parse_image_urls,
)


def test_guess_image_column():
    assert guess_image_column(["SKU", "Product_Image_URL", "Price"]) == "Product_Image_URL"


def test_guess_price_column():
    assert guess_price_column(["item", "unit_cost"]) == "unit_cost"


def test_guess_description_column():
    assert guess_description_column(["id", "product_title"]) == "product_title"


def test_parse_image_urls_comma_separated():
    cell = (
        "https://a.com/1.jpeg, https://b.com/2.jpeg"
    )
    u = parse_image_urls(cell)
    assert len(u) == 2
    assert u[0].startswith("https://a.com")


def test_parse_image_urls_empty():
    assert parse_image_urls("") == []
    assert parse_image_urls(float("nan")) == []
