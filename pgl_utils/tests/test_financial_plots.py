import pytest
import pandas as pd
import numpy as np
from pgl_utils.genai import financial_plots


def test_plot_raw_data():
    data = pd.Series([1, 2, 3, 4, 5])
    log_returns = np.array([0.01, -0.02, 0.015, -0.005, 0.007])
    financial_plots.plot_raw_data(data, log_returns, 'AAPL')


def test_plot_learning_curve():
    loss_history = [0.99, 0.85, 0.75, 0.65, 0.55]
    financial_plots.plot_learning_curve(loss_history)


def test_plot_projections_and_distributions():
    close_prices = pd.Series(np.linspace(100, 110, 300))
    log_returns = np.random.normal(0, 0.01, 300)
    generated_paths = [np.linspace(100, 120, 300), np.linspace(100, 115, 300)]
    financial_plots.plot_projections_and_distributions(close_prices, log_returns, generated_paths, 'AAPL')


def test_plot_individual_paths():
    close_prices = pd.Series(np.linspace(100, 110, 300))
    generated_paths = [np.linspace(100, 120, 300), np.linspace(100, 115, 300)]
    financial_plots.plot_individual_paths(close_prices, generated_paths, 'AAPL')


def test_plot_return_statistics():
    log_returns = np.random.normal(0, 0.01, 300)
    generated_paths = [np.linspace(100, 120, 300), np.linspace(100, 115, 300)]
    financial_plots.plot_return_statistics(log_returns, generated_paths)


def test_plot_volatility_cone():
    close_prices = pd.Series(np.linspace(100, 110, 300))
    fan_paths = [np.linspace(100, 120, 300) for _ in range(50)]
    financial_plots.plot_volatility_cone(close_prices, fan_paths, 'AAPL', 50)

def main():
    import pytest
    pytest.main()

if __name__ == "__main__":
    main()

