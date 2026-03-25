from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalyzerConfig:
    """Tunable defaults; CLI can override."""

    # Cosine similarity in [0, 1] for CLIP embeddings (higher = stricter duplicate).
    image_similarity_threshold: float = 0.86
    # How many neighbors to check per row (speed vs recall).
    image_neighbor_k: int = 15
    # Price z-score above this counts as a price outlier (when enough numeric rows).
    price_zscore_threshold: float = 3.5
    # Minimum description length (characters) before flagging.
    min_description_length: int = 5
    request_timeout_sec: float = 20.0
    max_image_bytes: int = 8 * 1024 * 1024
    cache_dir: Path = Path(".retail_analyzer_cache")
