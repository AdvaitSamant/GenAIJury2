"""
Evaluation Package
Includes comprehensive multi-class metrics (Top-k, Macro/Weighted F1, MCC, ECE, Brier Score)
and production-ready metric visualizers.
"""

from .metrics_and_plots import (
    compute_comprehensive_metrics,
    calculate_top_k_accuracy,
    calculate_ece,
    plot_training_curves,
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_class_performance,
    plot_metrics_summary,
)

__all__ = [
    "compute_comprehensive_metrics",
    "calculate_top_k_accuracy",
    "calculate_ece",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_calibration_curve",
    "plot_class_performance",
    "plot_metrics_summary",
]
