"""
Build Excel + PNG fixtures: three synthetic SKUs at different rotations, with
closely related descriptions within each SKU (27 rows total). Used to validate
CLIP grouping + merge UX on both tight duplicate clusters and a larger sheet.

Run from repo root:
  python scripts/generate_rotated_test_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retail_analyzer.excel_io import save_excel
OUT_DIR = ROOT / "tests" / "fixtures" / "rotated_duplicate_dataset"
EXCEL = OUT_DIR / "rotated_duplicates_sample.xlsx"


def _make_base(size: int = 280) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(245, 240, 230))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((20, 20, size - 20, size - 20), radius=24, fill=(34, 99, 82), outline=(15, 60, 50), width=3)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    text = "SKU-ROTMIX-01"
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size - tw) // 2, (size - th) // 2 - 20), text, fill=(240, 253, 244), font=font)
    d.text((size // 2 - 40, size // 2 + 20), "400g", fill=(204, 251, 241), font=font)
    return img


def _make_other_product(size: int = 280) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(30, 27, 75))
    d = ImageDraw.Draw(img)
    d.ellipse((40, 40, size - 40, size - 40), fill=(67, 56, 202), outline=(165, 180, 252), width=4)
    d.text((size // 2 - 50, size // 2 - 10), "TEA-99", fill=(238, 242, 255))
    return img


def _make_oat_drink(size: int = 280) -> Image.Image:
    """Third distinct SKU — tall carton, should not group with A or B."""
    img = Image.new("RGB", (size, size), color=(250, 248, 240))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((70, 40, size - 70, size - 50), radius=12, fill=(234, 179, 8), outline=(161, 98, 7), width=3)
    d.text((size // 2 - 55, size // 2 - 25), "OAT-1L", fill=(66, 32, 6))
    d.text((size // 2 - 35, size // 2 + 5), "barista", fill=(120, 53, 15))
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _make_base()
    other = _make_other_product()
    oat = _make_oat_drink()

    # Product A: 4 rotations (same item); rows below cycle these paths
    angles_a = (0, 72, 144, 216)
    paths_a: list[str] = []
    for deg in angles_a:
        rot = base.rotate(-deg, expand=True, fillcolor=(245, 240, 230))
        p = OUT_DIR / f"product_a_rot_{deg}.png"
        rot.save(p, format="PNG")
        paths_a.append(str(p.relative_to(ROOT)).replace("\\", "/"))

    # Product B: different item, 4 angles (should not group with A)
    angles_b = (0, 90, 180, 270)
    paths_b: list[str] = []
    for deg in angles_b:
        rot = other.rotate(-deg, expand=True, fillcolor=(30, 27, 75))
        p = OUT_DIR / f"product_b_rot_{deg}.png"
        rot.save(p, format="PNG")
        paths_b.append(str(p.relative_to(ROOT)).replace("\\", "/"))

    # Product C: oat drink — two rotations to mirror messy catalog uploads
    angles_c = (0, 45)
    paths_c: list[str] = []
    for deg in angles_c:
        rot = oat.rotate(-deg, expand=True, fillcolor=(250, 248, 240))
        p = OUT_DIR / f"product_c_rot_{deg}.png"
        rot.save(p, format="PNG")
        paths_c.append(str(p.relative_to(ROOT)).replace("\\", "/"))

    desc_a = [
        "Organic trail mix 400g — nuts and dried fruit blend",
        "Trail mix 400g organic: dried fruit & mixed nuts",
        "400g organic trail mix with nuts, dried fruit (same SKU)",
        "Dried fruit and nut trail mix, 400 gram organic",
        "ORG TRAIL MIX 400G - nuts, raisins, cranberries",
        "Trail mix (organic) 400 g multipack nuts & fruit",
        "Nuts and dried fruit mix, organic, four hundred grams",
        "400g pack: organic trail mix, nut and fruit blend",
        "Store brand organic trail mix — 400g resealable bag",
        "Bulk aisle: trail mix organic 400g fruit & nut",
        "Resealable organic trail mix, 400g, nuts and fruit",
        "On promo: organic trail mix 400g dried fruit mix",
    ]
    desc_b = [
        "Herbal infusion tea assortment 20 bags calming blend",
        "Calming herbal tea box — 20 tea bags assorted",
        "Assorted herbal teas 20pk — calming infusion variety",
        "20-count herbal tea bags, mixed calming flavors",
        "Tea gift set: 20 herbal infusion sachets",
        "Wellness herbal tea assortment, twenty tea bags",
        "Mixed herbal tea 20 bags (chamomile, mint, lemon)",
    ]
    desc_c = [
        "Oat drink barista edition 1 liter carton",
        "1L barista oat milk — foamable, unsweetened",
        "Unsweetened oat beverage 1L for coffee and baking",
        "Plant-based oat drink 1000ml barista style",
        "Oat milk alternative 1 litre — coffee shop blend",
        "Barista oat 1L: creamy, steams well for lattes",
        "1 liter oat drink, barista formula, dairy-free",
        "Oat beverage carton 1L unsweetened barista",
    ]

    rows: list[dict] = []
    n_a = len(desc_a)
    for i in range(n_a):
        rows.append(
            {
                "product_id": f"SKU-A-{i:02d}",
                "product_name": "Organic Trail Mix 400g",
                "product_description": desc_a[i],
                "unit_price": 8.99,
                "image_urls": paths_a[i % len(paths_a)],
            }
        )
    n_b = len(desc_b)
    for i in range(n_b):
        rows.append(
            {
                "product_id": f"SKU-B-{i:02d}",
                "product_name": "Herbal Tea Assorted 20ct",
                "product_description": desc_b[i],
                "unit_price": 5.49,
                "image_urls": paths_b[i % len(paths_b)],
            }
        )
    n_c = len(desc_c)
    for i in range(n_c):
        rows.append(
            {
                "product_id": f"SKU-C-{i:02d}",
                "product_name": "Barista Oat Drink 1L",
                "product_description": desc_c[i],
                "unit_price": 3.29,
                "image_urls": paths_c[i % len(paths_c)],
            }
        )

    df = pd.DataFrame(rows)
    save_excel(df, EXCEL, image_column="image_urls")
    print(f"Wrote {EXCEL}")
    print(f"Images under {OUT_DIR}")


if __name__ == "__main__":
    main()
