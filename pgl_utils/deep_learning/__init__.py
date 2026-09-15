"""
Deep Learning utilities
"""

from .architectures import draw_neural_network
from .plots import (
    plot_time_series,
    plot_two_series_comparison,
    plot_series_with_markers,
    plot_histogram_with_normal_curve,
    plot_sliding_window,
    plot_full_sliding_progress,
    format_price_axis,
    plot_real_vs_synthetic_continuation,
    plot_paths_grid,
    plot_loss_curve,
    plot_reconstruction_error_with_threshold,
    plot_outlier_detection_and_trading_signals,
)
from .tickers import load_brazil_tickers, load_us_tickers
from .trading import TradingSignalGenerator

__all__ = [
    "draw_neural_network",
    "plot_time_series",
    "plot_two_series_comparison",
    "plot_series_with_markers",
    "plot_histogram_with_normal_curve",
    "plot_sliding_window",
    "plot_full_sliding_progress",
    "format_price_axis",
    "plot_real_vs_synthetic_continuation",
    "plot_paths_grid",
    "plot_loss_curve",
    "plot_reconstruction_error_with_threshold",
    "plot_outlier_detection_and_trading_signals",
    "load_brazil_tickers",
    "load_us_tickers",
    "TradingSignalGenerator",
]
