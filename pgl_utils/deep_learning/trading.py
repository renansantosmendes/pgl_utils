"""
Rule-based trading signal generation from anomaly detection outputs.
"""

from __future__ import annotations

import numpy as np


class TradingSignalGenerator:
    """Rule-based generator that turns anomaly flags into trading
    signals.

    The generator inspects the short-term price momentum around every
    point flagged as an outlier by an upstream anomaly detector (for
    example, the reconstruction error of an autoencoder) and converts
    each anomaly into a "buy", "sell" or "hold" signal.

    Attributes:
        momentum_window: The number of trading days used to compute
            the short-term momentum around each anomaly.
        neutral_zone: The minimum absolute momentum, in log-return
            terms, required for an anomaly to generate a buy or sell
            signal instead of "hold".
    """

    def __init__(
        self,
        momentum_window: int = 3,
        neutral_zone: float = 0.01,
    ) -> None:
        """Initialize the generator with its decision parameters.

        Args:
            momentum_window: The number of trading days used to
                compute the short-term momentum around each anomaly.
            neutral_zone: The minimum absolute momentum required for
                an anomaly to generate a buy or sell signal.
        """
        self.momentum_window = momentum_window
        self.neutral_zone = neutral_zone

    def generate_signals(
        self,
        price_series: np.ndarray,
        is_outlier: np.ndarray,
    ) -> np.ndarray:
        """Generate a trading signal for every observation.

        Args:
            price_series: A one-dimensional array with the asset
                price for every observation in the dataset.
            is_outlier: A boolean array, with the same length as
                `price_series`, indicating whether each observation
                was flagged as an anomaly by the detector.

        Returns:
            An array of strings, with the same length as the inputs,
            containing "buy", "sell" or "hold" for every observation.

        Example:
            >>> generator = TradingSignalGenerator()
            >>> signals = generator.generate_signals(
            ...     df["price"].values, df["is_outlier"].values,
            ... )
        """
        number_of_observations = len(price_series)
        signals = np.full(number_of_observations, "hold", dtype=object)

        for index in np.where(is_outlier)[0]:
            window_start = max(0, index - self.momentum_window)
            if window_start == index:
                continue

            momentum = (
                price_series[index] - price_series[window_start]
            ) / price_series[window_start]

            if momentum < -self.neutral_zone:
                signals[index] = "buy"
            elif momentum > self.neutral_zone:
                signals[index] = "sell"

        return signals
