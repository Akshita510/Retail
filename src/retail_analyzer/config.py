from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalyzerConfig:
    """Tunable defaults; CLI can override."""

    request_timeout_sec: float = 20.0
    max_image_bytes: int = 8 * 1024 * 1024
    cache_dir: Path = Path(".retail_analyzer_cache")

    # Duplicate detection: same image URL + sentence-embedding similarity
    duplicate_desc_cosine_threshold: float = 0.8
    duplicate_category_cosine_threshold: float = 0.8
    same_image_desc_anomaly_threshold: float = 0.4

    # Price outlier within category (mean ± n * std on raw positive prices)
    min_category_group_size: int = 5
    price_outlier_std_multiplier: float = 3.0

    # Legacy name kept for API compatibility (unused for URL-based duplicates)
    image_similarity_threshold: float = 0.86
    image_neighbor_k: int = 15
    price_zscore_threshold: float = 3.5
    min_description_length: int = 5
    enable_context_anomalies: bool = True
    context_typo_ratio_lo: float = 9.0
    context_typo_ratio_hi: float = 11.5
    context_dimension_rel_tolerance: float = 0.18
    context_price_text_mismatch_ratio: float = 3.0
    context_max_spec_cell_chars: int = 400
    price_category_zscore_threshold: float = 3.0
    min_text_len_for_embedding: int = 15
    desc_category_cosine_threshold: float = 0.28
    desc_within_category_sim_threshold: float = 0.40
    min_category_text_group_size: int = 4
