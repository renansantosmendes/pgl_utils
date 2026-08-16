"""
Plotting utilities for time series / tensor practice notebooks.
"""

from __future__ import annotations

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
