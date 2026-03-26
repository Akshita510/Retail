"""
Retail catalog anomalies: re-exports the retail anomaly detector.

See ``retail_anomaly_detection`` for ``anomaly_type`` values and rules.
"""

from retail_analyzer.retail_anomaly_detection import ANOMALY_HELP_BULLETS, detect_anomalies

__all__ = ["detect_anomalies", "ANOMALY_HELP_BULLETS"]
