# Retail catalog analyzer

A Python toolkit for **retail product spreadsheets** (Excel `.xlsx`). It helps you find **visually similar listings** that may be duplicates (same product, different rows) and **data anomalies** (mismatches between images, text, and prices), then **export** results back to Excel—with optional **image thumbnails** next to URL columns in downloaded files.

---

## What it does

### Duplication analysis

- Reads your catalog from the **first sheet** of an uploaded workbook.
- Detects an **image column** from headers (names containing *image*, *photo*, *url*, *link*, etc.).
- Downloads remote images (or loads local file paths), embeds them with a **CLIP**-style model, and groups rows whose images are **similar** above a configurable threshold.
- Adds columns such as:
  - `similar_image_group_id` / `duplicate_group_id` — rows in the same group are candidates to merge.
  - `similarity_score`, `similarity_comment` — how strong the match is and a short explanation.
- In the **Streamlit** app you can **merge** duplicate groups into one row per group and download **merged** Excel.

### Anomaly analysis

- Runs a separate set of **rule-based and embedding-based checks** (e.g. image–description mismatch, price outliers within categories).
- Flags rows with `anomaly_flag`, `anomaly_type`, `reason`, and related scores—suited for manual review.

### Excel exports

- **Downloaded** workbooks (from the dashboard or CLI output) can include:
  - **Hyperlinks** on image URL cells (click to open the file or web URL).
  - An **Image preview** column with **embedded thumbnails** for the first image in each cell (Excel cannot show a live picture *inside* the URL cell’s hover tooltip; the preview column is the reliable way to see the image beside the link).

---

## Requirements

- **Python 3.10+**
- **Poetry** — dependency and virtual environment management.

Heavy dependencies include **PyTorch**, **sentence-transformers**, **pandas**, **openpyxl**, and **Streamlit**. First install may take a while while wheels download.

---

## Install

```bash
cd Retail_project
poetry install
```

Activate the environment (optional, depending on how you run commands):

```bash
poetry shell
```

---

## Run the web dashboard

```bash
poetry run streamlit run dashboard.py
```

1. Choose **Duplication analysis** or **Anomaly analysis** (one mode at a time).
2. Upload your `.xlsx` (first sheet is used).
3. Adjust options (e.g. similarity threshold for duplication) and run analysis.
4. Review tables, edit the sheet in the UI if needed, and use **Download** to export the current sheet.  
   For duplication, you can open **Merge duplicate groups** to build a merged catalog and download it.

---

## Command-line usage

The package exposes a Typer CLI:

```bash
poetry run retail-analyze INPUT.xlsx --output OUTPUT.xlsx
```

Useful options:

| Option | Meaning |
|--------|--------|
| `--sheet` | Sheet index or name (default `0`). |
| `--image-column`, `--price-column`, `--description-column` | Override auto-detected columns. |
| `--skip-images` | Skip image download / similarity (anomaly-oriented run only). |
| `--similarity-threshold` | CLIP cosine threshold for grouping (default `0.86`). |
| `--neighbor-k` | How many neighbors to compare per image (default `15`). |
| `--cache-dir` | Where downloaded images are cached (default `.retail_analyzer_cache`). |
| `-o` / `--output` | Write a highlighted workbook with analysis columns. |

Example:

```bash
poetry run retail-analyze catalog.xlsx -o catalog_analyzed.xlsx --similarity-threshold 0.88
```

---

## Test data: rotated duplicate fixture

A small synthetic dataset (same product at different rotations + different SKUs) lives under `tests/fixtures/rotated_duplicate_dataset/`. Regenerate it with:

```bash
poetry run python scripts/generate_rotated_test_dataset.py
```

This writes `rotated_duplicates_sample.xlsx` and PNG assets. Use it to sanity-check duplication grouping and exports without real catalog data.

---

## Run tests

```bash
poetry run pytest
```

---

## Project layout (brief)

| Path | Role |
|------|------|
| `dashboard.py` | Streamlit UI for upload, analysis, merge, and downloads. |
| `src/retail_analyzer/` | Core library: pipeline, duplicate detection, anomalies, Excel I/O, merge, image fetch/cache, optional Excel image previews. |
| `scripts/` | Helper scripts (e.g. rotated test dataset generator). |
| `tests/` | Pytest suite and fixtures. |

---

## Configuration and cache

- **Cache directory** (default `.retail_analyzer_cache/`) stores downloaded images so repeated runs are faster. Safe to delete; it will be recreated.
- Tunable defaults live in `AnalyzerConfig` (`src/retail_analyzer/config.py`). The dashboard and CLI pass relevant options when you change sliders or flags.

---

## License

See project maintainers for license terms if not specified in this repository.
