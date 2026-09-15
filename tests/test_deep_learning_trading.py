import numpy as np

from pgl_utils.deep_learning.trading import TradingSignalGenerator


def test_generate_signals_returns_hold_when_no_outliers():
    price_series = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
    is_outlier = np.zeros(len(price_series), dtype=bool)

    generator = TradingSignalGenerator()
    signals = generator.generate_signals(price_series, is_outlier)

    assert list(signals) == ["hold"] * len(price_series)


def test_generate_signals_flags_buy_on_price_drop():
    price_series = np.array([10.0, 10.0, 10.0, 8.0, 8.0])
    is_outlier = np.array([False, False, False, True, False])

    generator = TradingSignalGenerator(momentum_window=3, neutral_zone=0.01)
    signals = generator.generate_signals(price_series, is_outlier)

    assert signals[3] == "buy"


def test_generate_signals_flags_sell_on_price_rise():
    price_series = np.array([10.0, 10.0, 10.0, 12.0, 12.0])
    is_outlier = np.array([False, False, False, True, False])

    generator = TradingSignalGenerator(momentum_window=3, neutral_zone=0.01)
    signals = generator.generate_signals(price_series, is_outlier)

    assert signals[3] == "sell"


def test_generate_signals_holds_inside_neutral_zone():
    price_series = np.array([10.0, 10.0, 10.0, 10.02, 10.02])
    is_outlier = np.array([False, False, False, True, False])

    generator = TradingSignalGenerator(momentum_window=3, neutral_zone=0.01)
    signals = generator.generate_signals(price_series, is_outlier)

    assert signals[3] == "hold"


def test_generate_signals_ignores_outlier_at_series_start():
    price_series = np.array([10.0, 9.0, 8.0])
    is_outlier = np.array([True, False, False])

    generator = TradingSignalGenerator(momentum_window=3, neutral_zone=0.01)
    signals = generator.generate_signals(price_series, is_outlier)

    assert signals[0] == "hold"
