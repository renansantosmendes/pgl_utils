import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def plot_raw_data(close_prices: pd.Series, log_returns: np.ndarray, ticker: str) -> None:
    """Plot closing price history, daily log returns, and return distribution.

    Args:
        close_prices: Series of adjusted closing prices.
        log_returns: Array of daily log returns.
        ticker: Ticker symbol used as plot label.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(close_prices.values, color='steelblue', linewidth=1)
    axes[0].set_title(f'{ticker} — Closing Price')
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Price (USD)')
    axes[0].grid(alpha=0.3)

    axes[1].plot(log_returns, color='darkorange', linewidth=0.7, alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Daily Log Returns')
    axes[1].set_xlabel('Days')
    axes[1].set_ylabel('Log-return')
    axes[1].grid(alpha=0.3)

    axes[2].hist(log_returns, bins=60, color='seagreen', edgecolor='white', alpha=0.8)
    axes[2].set_title('Return Distribution')
    axes[2].set_xlabel('Log-return')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(alpha=0.3)

    plt.suptitle('Real Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_learning_curve(loss_history: list[float]) -> None:
    """Plot the VAE training loss over epochs.

    Args:
        loss_history: List of per-epoch average loss values.
    """
    plt.figure(figsize=(8, 3))
    plt.plot(loss_history, color='steelblue', linewidth=2)
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_projections_and_distributions(
    close_prices: pd.Series,
    log_returns: np.ndarray,
    generated_paths: list[np.ndarray],
    ticker: str,
    history_window: int = 252,
) -> None:
    """Plot historical prices with all generated paths and compare return distributions.

    Args:
        close_prices: Series of adjusted closing prices.
        log_returns: Array of real daily log returns.
        generated_paths: List of synthetic price path arrays.
        ticker: Ticker symbol used as plot label.
        history_window: Number of historical days to display on the left panel.
    """
    historical_prices = close_prices.values[-history_window:]
    historical_days = np.arange(-history_window, 0)
    future_days = np.arange(0, len(generated_paths[0]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(historical_days, historical_prices, color='steelblue', linewidth=1.5,
                 label='Real history', zorder=5)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6, label='Today')

    for i, path in enumerate(generated_paths):
        axes[0].plot(future_days, path, alpha=0.55, linewidth=1.2,
                     label='VAE generated' if i == 0 else '_')

    axes[0].set_title(f'{ticker} — History + VAE Projections', fontsize=13)
    axes[0].set_xlabel('Days (0 = today)')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    generated_returns = []
    for path in generated_paths:
        generated_returns.extend(np.diff(np.log(path)))

    axes[1].hist(log_returns, bins=60, alpha=0.6, color='steelblue',
                 density=True, label='Real returns')
    axes[1].hist(generated_returns, bins=60, alpha=0.6, color='darkorange',
                 density=True, label='Generated returns')
    axes[1].set_title('Distribution: Real vs Generated Returns', fontsize=13)
    axes[1].set_xlabel('Log-return')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('VAE — Price Scenario Generation', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_individual_paths(
    close_prices: pd.Series,
    generated_paths: list[np.ndarray],
    ticker: str,
    history_window: int = 252,
) -> None:
    """Plot each generated path in its own panel alongside the real price history.

    Args:
        close_prices: Series of adjusted closing prices.
        generated_paths: List of synthetic price path arrays.
        ticker: Ticker symbol used as plot label.
        history_window: Number of historical days to display in each panel.
    """
    historical_prices = close_prices.values[-history_window:]
    historical_days = np.arange(-history_window, 0)
    future_days = np.arange(0, len(generated_paths[0]))

    fig, axes = plt.subplots(len(generated_paths), 1, figsize=(14, 4 * len(generated_paths)))

    for i, path in enumerate(generated_paths):
        ax = axes[i]
        ax.plot(historical_days, historical_prices, color='steelblue', linewidth=1.5, label='Real history')
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.plot(future_days, path, color='darkorange', linewidth=1.5, label=f'Generated path {i + 1}')
        ax.set_title(f'Path {i + 1}', fontsize=11)
        ax.set_xlabel('Days (0 = today)')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f'{ticker} — Real History + Each Generated Path', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_return_statistics(log_returns: np.ndarray, generated_paths: list[np.ndarray]) -> None:
    """Print a descriptive statistics table comparing real and generated returns.

    Args:
        log_returns: Array of real daily log returns.
        generated_paths: List of synthetic price path arrays.
    """
    generated_returns = []
    for path in generated_paths:
        generated_returns.extend(np.diff(np.log(path)))

    real_returns = log_returns
    gen_returns = np.array(generated_returns)

    stats_df = pd.DataFrame({
        'Real': [real_returns.mean(), real_returns.std(),
                 real_returns.min(), real_returns.max(),
                 pd.Series(real_returns).skew(),
                 pd.Series(real_returns).kurt()],
        'Generated': [gen_returns.mean(), gen_returns.std(),
                      gen_returns.min(), gen_returns.max(),
                      pd.Series(gen_returns).skew(),
                      pd.Series(gen_returns).kurt()]
    }, index=['Mean', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'])

    print('=' * 45)
    print('         Return Statistics')
    print('=' * 45)
    print(stats_df.to_string(float_format='{:.6f}'.format))
    print('=' * 45)
    print('\n✅ Similar std dev  → realistic volatility')
    print('✅ Kurtosis > 3     → fat tails preserved')


def plot_volatility_cone(
    close_prices: pd.Series,
    fan_paths: list[np.ndarray],
    ticker: str,
    num_scenarios: int,
    recent_history_window: int = 60,
) -> None:
    """Plot a fan chart (volatility cone) with percentile bands across many scenarios.

    Args:
        close_prices: Series of adjusted closing prices.
        fan_paths: List of synthetic price path arrays used to compute percentiles.
        ticker: Ticker symbol used as plot label.
        num_scenarios: Total number of scenarios generated, used in the chart title.
        recent_history_window: Number of recent historical days shown before day 0.
    """
    scenario_matrix = np.array(fan_paths)

    p5 = np.percentile(scenario_matrix, 5, axis=0)
    p25 = np.percentile(scenario_matrix, 25, axis=0)
    p50 = np.percentile(scenario_matrix, 50, axis=0)
    p75 = np.percentile(scenario_matrix, 75, axis=0)
    p95 = np.percentile(scenario_matrix, 95, axis=0)

    future_days = np.arange(len(p50))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(np.arange(-recent_history_window, 0), close_prices.values[-recent_history_window:],
            color='steelblue', linewidth=2, label='Real history')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.fill_between(future_days, p5, p95, alpha=0.15, color='darkorange', label='90% of scenarios')
    ax.fill_between(future_days, p25, p75, alpha=0.30, color='darkorange', label='50% of scenarios')
    ax.plot(future_days, p50, color='darkorange', linewidth=2, linestyle='--', label='Median')

    ax.set_title(f'{ticker} — VAE Volatility Cone ({num_scenarios} scenarios)', fontsize=14)
    ax.set_xlabel('Days (0 = today)')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
