"""
Plotting utilities for time series / tensor practice notebooks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader


def plot_time_series(
    series_data: pd.DataFrame,
    column_name: str,
    chart_title: str,
) -> go.Figure:
    """Plot a time series column using an interactive Plotly line chart."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=series_data.index,
            y=series_data[column_name],
            mode="lines",
            name=column_name,
        )
    )
    figure.update_layout(
        title=chart_title,
        xaxis_title="Data",
        yaxis_title="Preço de Fechamento (USD)",
        template="plotly_white",
    )
    return figure


def plot_two_series_comparison(
    x_axis_values: np.ndarray,
    first_series: np.ndarray,
    second_series: np.ndarray,
    first_series_name: str,
    second_series_name: str,
    chart_title: str,
) -> go.Figure:
    """Plot two series sharing the same x-axis on top of each other for comparison."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_axis_values,
            y=first_series,
            mode="lines",
            name=first_series_name,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_axis_values,
            y=second_series,
            mode="lines",
            name=second_series_name,
        )
    )
    figure.update_layout(
        title=chart_title,
        xaxis_title="Índice temporal",
        yaxis_title="Valor",
        template="plotly_white",
    )
    return figure


def plot_series_with_markers(
    x_axis_values: np.ndarray,
    line_series: np.ndarray,
    marker_x_values: np.ndarray,
    marker_y_values: np.ndarray,
    line_series_name: str,
    marker_series_name: str,
    chart_title: str,
) -> go.Figure:
    """Plot a line series and highlight specific points on top of it with markers."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_axis_values,
            y=line_series,
            mode="lines",
            name=line_series_name,
            line=dict(color="lightgray"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=marker_x_values,
            y=marker_y_values,
            mode="markers",
            name=marker_series_name,
            marker=dict(size=8, color="orange"),
        )
    )
    figure.update_layout(
        title=chart_title,
        xaxis_title="Índice temporal",
        yaxis_title="Valor",
        template="plotly_white",
    )
    return figure


def plot_histogram_with_normal_curve(
    sample_values: torch.Tensor,
    number_of_bins: int,
    chart_title: str,
) -> go.Figure:
    """Plot a normalized histogram of a tensor alongside its fitted normal distribution curve."""
    sample_mean = sample_values.mean()
    sample_std = sample_values.std()
    x_grid = torch.linspace(sample_values.min(), sample_values.max(), 200)
    normal_pdf = (1.0 / (sample_std * torch.sqrt(torch.tensor(2.0 * np.pi)))) * torch.exp(
        -0.5 * ((x_grid - sample_mean) / sample_std) ** 2
    )

    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=sample_values.numpy(),
            nbinsx=number_of_bins,
            histnorm="probability density",
            name="Distribuição empírica",
            opacity=0.7,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_grid.numpy(),
            y=normal_pdf.numpy(),
            mode="lines",
            name="Normal ajustada N(mean, std)",
            line=dict(color="orange", width=3),
        )
    )
    figure.update_layout(
        title=chart_title,
        xaxis_title="Valor",
        yaxis_title="Densidade",
        template="plotly_white",
    )
    return figure


def plot_sliding_window(
    full_series: np.ndarray,
    input_window: torch.Tensor,
    output_window: torch.Tensor,
    window_start_index: int,
    chart_title: str,
) -> go.Figure:
    """Plot the full series alongside a highlighted input/output sliding window."""
    time_axis = np.arange(len(full_series))
    input_axis = np.arange(window_start_index, window_start_index + len(input_window))
    output_axis = np.arange(
        window_start_index + len(input_window),
        window_start_index + len(input_window) + len(output_window),
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=time_axis,
            y=full_series,
            mode="lines",
            name="Série completa",
            line=dict(color="lightgray"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=input_axis,
            y=input_window.numpy(),
            mode="lines+markers",
            name="Janela de entrada",
            line=dict(color="royalblue"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=output_axis,
            y=output_window.numpy(),
            mode="lines+markers",
            name="Janela de saída",
            line=dict(color="orange"),
        )
    )
    figure.update_layout(
        title=chart_title,
        xaxis_title="Índice temporal",
        yaxis_title="Preço de Fechamento (USD)",
        template="plotly_white",
    )
    return figure


def plot_full_sliding_progress(
    full_series: np.ndarray,
    dataloader: DataLoader,
    input_window_size: int,
    max_windows_to_plot: int,
) -> go.Figure:
    """Overlay multiple sliding input windows on top of the full time series."""
    time_axis = np.arange(len(full_series))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=time_axis,
            y=full_series,
            mode="lines",
            name="Série completa",
            line=dict(color="lightgray"),
        )
    )

    window_counter = 0
    for batch_index, (batch_input, _) in enumerate(dataloader):
        for sample_index in range(batch_input.shape[0]):
            if window_counter >= max_windows_to_plot:
                figure.update_layout(
                    title="Deslizamento da janela de entrada sobre a série temporal",
                    xaxis_title="Índice temporal",
                    yaxis_title="Preço de Fechamento (USD)",
                    template="plotly_white",
                )
                return figure

            window_start_index = batch_index * dataloader.batch_size + sample_index
            window_axis = np.arange(window_start_index, window_start_index + input_window_size)
            figure.add_trace(
                go.Scatter(
                    x=window_axis,
                    y=batch_input[sample_index].numpy(),
                    mode="lines",
                    name=f"Janela {window_counter}",
                    opacity=0.6,
                    showlegend=False,
                )
            )
            window_counter += 1

    figure.update_layout(
        title="Deslizamento da janela de entrada sobre a série temporal",
        xaxis_title="Índice temporal",
        yaxis_title="Preço de Fechamento (USD)",
        template="plotly_white",
    )
    return figure


def format_price_axis(axis: plt.Axes) -> None:
    """Format a matplotlib axis to display values as US dollars."""
    axis.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"US$ {value:,.2f}")
    )


def plot_real_vs_synthetic_continuation(
    return_dates: pd.DatetimeIndex,
    log_returns: np.ndarray,
    price_index: pd.DatetimeIndex,
    price_values: np.ndarray,
    synthetic_dates: pd.DatetimeIndex,
    synthetic_returns: np.ndarray,
    synthetic_prices: np.ndarray,
    mean_synthetic_return_path: np.ndarray,
    mean_synthetic_price_path: np.ndarray,
    ticker_symbol: str,
    n_paths_to_plot: int,
    output_path: str | None = None,
) -> plt.Figure:
    """Plot the real series followed by its synthetic continuation, in log-return and price panels."""
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)

    axes[0].plot(
        return_dates, log_returns, color="royalblue", linewidth=0.9,
        label="Log-retorno real",
    )
    for path in synthetic_returns[:n_paths_to_plot]:
        axes[0].plot(synthetic_dates, path, color="tab:orange", alpha=0.15, linewidth=0.8)
    axes[0].plot(
        synthetic_dates, mean_synthetic_return_path, color="tab:orange",
        linewidth=1.8, label="Log-retorno sintético (média dos caminhos)",
    )
    axes[0].axvline(return_dates[-1], color="gray", linestyle="--", linewidth=1)
    axes[0].set_title(f"{ticker_symbol} — Log-retorno: real + continuação sintética")
    axes[0].set_ylabel("Log-retorno")
    axes[0].legend()

    axes[1].plot(price_index, price_values, color="royalblue", linewidth=1.2, label="Preço real")
    for path in synthetic_prices[:n_paths_to_plot]:
        axes[1].plot(synthetic_dates, path, color="tab:orange", alpha=0.15, linewidth=0.8)
    axes[1].plot(
        synthetic_dates, mean_synthetic_price_path, color="tab:orange",
        linewidth=1.8, label="Preço sintético (média dos caminhos)",
    )
    axes[1].axvline(
        return_dates[-1], color="gray", linestyle="--", linewidth=1,
        label="Fim da série real / início da sintética",
    )
    axes[1].set_title(f"{ticker_symbol} — Preço: real + continuação sintética")
    axes[1].set_ylabel("Preço")
    format_price_axis(axes[1])
    axes[1].legend()

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=130)
    plt.show()
    return figure


def plot_paths_grid(
    real_tail_dates: pd.DatetimeIndex,
    real_tail_values: np.ndarray,
    synthetic_dates: pd.DatetimeIndex,
    synthetic_paths: np.ndarray,
    n_paths: int,
    n_columns: int,
    title_prefix: str,
    y_axis_label: str,
    output_path: str,
    is_price: bool = False,
) -> None:
    """Plot a grid with one synthetic path per panel.

    Each panel shows a recent slice of the real series immediately
    followed by a single synthetic path, sharing the same time axis, to
    make it easier to visually compare paths one at a time.

    Args:
        real_tail_dates: Dates corresponding to the recent slice of the
            real series to display in every panel.
        real_tail_values: Real series values aligned with
            `real_tail_dates`.
        synthetic_dates: Dates corresponding to the synthetic horizon.
        synthetic_paths: Two-dimensional array of shape
            (n_available_paths, horizon_days) with the synthetic paths.
        n_paths: Number of synthetic paths to plot, one per panel.
        n_columns: Number of columns in the grid.
        title_prefix: Title shown at the top of the whole figure.
        y_axis_label: Label used for the y-axis of the leftmost panels.
        output_path: File path where the resulting figure is saved.
        is_price: Whether the values represent prices, in which case the
            y-axis is formatted as US dollars. Defaults to False.

    Returns:
        None. The figure is saved to `output_path` and displayed inline.

    Example:
        >>> plot_paths_grid(
        ...     real_tail_dates=return_dates[-60:],
        ...     real_tail_values=log_returns[-60:],
        ...     synthetic_dates=synthetic_dates,
        ...     synthetic_paths=synthetic_returns,
        ...     n_paths=9,
        ...     n_columns=3,
        ...     title_prefix="AAPL — Log-retorno sintético",
        ...     y_axis_label="Log-retorno",
        ...     output_path="outputs/grid.png",
        ... )
    """
    n_paths = min(n_paths, synthetic_paths.shape[0])
    n_rows = int(np.ceil(n_paths / n_columns))

    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(4.2 * n_columns, 3.2 * n_rows), sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for path_index in range(n_paths):
        axis = axes[path_index]
        axis.plot(real_tail_dates, real_tail_values, color="royalblue", linewidth=1.1, label="Real")
        axis.plot(
            synthetic_dates, synthetic_paths[path_index], color="tab:orange",
            linewidth=1.3, label="Sintético",
        )
        axis.axvline(real_tail_dates[-1], color="gray", linestyle="--", linewidth=0.8)
        axis.set_title(f"Caminho sintético {path_index + 1}", fontsize=10)
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        if is_price:
            format_price_axis(axis)
        if path_index % n_columns == 0:
            axis.set_ylabel(y_axis_label, fontsize=9)

    for empty_index in range(n_paths, len(axes)):
        figure.delaxes(axes[empty_index])

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=10)
    figure.suptitle(title_prefix, fontsize=13, y=1.06)
    figure.tight_layout()
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    
    
def plot_loss_curve(
    loss_history: list[float],
    chart_title: str,
    loss_series_name: str = "Loss de treino",
) -> go.Figure:
    """Plot a training loss curve across epochs."""
    epoch_axis = np.arange(1, len(loss_history) + 1)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=epoch_axis,
            y=loss_history,
            mode="lines+markers",
            name=loss_series_name,
            line=dict(color="royalblue"),
        )
    )
    figure.update_layout(
        title=chart_title,
        xaxis_title="Época",
        yaxis_title="Loss",
        template="plotly_white",
    )
    return figure
