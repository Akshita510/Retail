from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="Retail Excel: similar images + anomalies.")


@app.command()
def analyze(
    input_excel: Path = typer.Argument(..., exists=True, help="Input .xlsx path"),
    sheet: str = typer.Option(
        "0",
        help='Sheet index or name (e.g. "0" or "Catalog")',
    ),
    image_column: Optional[str] = typer.Option(
        None,
        help="Column with image URLs (auto-detected if omitted)",
    ),
    price_column: Optional[str] = typer.Option(
        None,
        help="Price / cost column (auto-detected if omitted)",
    ),
    description_column: Optional[str] = typer.Option(
        None,
        help="Description / title column (auto-detected if omitted)",
    ),
    skip_images: bool = typer.Option(
        False,
        "--skip-images",
        help="Run only anomaly rules (no image download / CLIP)",
    ),
    similarity_threshold: float = typer.Option(
        0.86,
        min=0.5,
        max=1.0,
        help="CLIP cosine similarity for same-product groups (higher = stricter)",
    ),
    neighbor_k: int = typer.Option(
        15,
        min=2,
        help="Image neighbors to check (higher = slower, better recall)",
    ),
    cache_dir: Path = typer.Option(
        Path(".retail_analyzer_cache"),
        help="Cache directory for downloaded images",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
    output_excel: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to write highlighted .xlsx",
    ),
) -> None:
    """Analyze a retail catalog Excel file."""
    from retail_analyzer.config import AnalyzerConfig
    from retail_analyzer.pipeline import run_analysis

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = AnalyzerConfig(
        image_similarity_threshold=similarity_threshold,
        image_neighbor_k=neighbor_k,
        cache_dir=cache_dir,
    )
    sheet_parsed: str | int = int(sheet) if sheet.isdigit() else sheet
    run_analysis(
        input_excel,
        output_path=output_excel,
        sheet=sheet_parsed,
        image_column=image_column,
        price_column=price_column,
        description_column=description_column,
        skip_images=skip_images,
        config=cfg,
    )
    if output_excel is not None:
        typer.echo(f"Wrote {output_excel}")
    else:
        typer.echo("Analysis complete (no file written; use --output to export .xlsx).")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
