import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader, TensorDataset

from pgl_utils.deep_learning.plots import (
    plot_time_series,
    plot_two_series_comparison,
    plot_series_with_markers,
    plot_histogram_with_normal_curve,
    plot_sliding_window,
    plot_full_sliding_progress,
)


def test_plot_time_series_returns_figure_with_expected_trace():
    series_data = pd.DataFrame({"close_price": [10.0, 11.0, 9.5, 12.0]})

    figure = plot_time_series(series_data, "close_price", "Test Title")

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    assert figure.data[0].mode == "lines"
    assert figure.data[0].name == "close_price"
    assert list(figure.data[0].y) == series_data["close_price"].tolist()
    assert figure.layout.title.text == "Test Title"


def test_plot_two_series_comparison_returns_both_series():
    x_axis_values = np.arange(5)
    first_series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    second_series = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    figure = plot_two_series_comparison(
        x_axis_values,
        first_series,
        second_series,
        "First",
        "Second",
        "Comparison",
    )

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert figure.data[0].name == "First"
    assert figure.data[1].name == "Second"
    assert list(figure.data[0].y) == first_series.tolist()
    assert list(figure.data[1].y) == second_series.tolist()
    assert figure.layout.title.text == "Comparison"


def test_plot_series_with_markers_returns_line_and_markers():
    x_axis_values = np.arange(10)
    line_series = np.arange(10, dtype=float)
    marker_x_values = np.array([2, 5, 8])
    marker_y_values = np.array([2.0, 5.0, 8.0])

    figure = plot_series_with_markers(
        x_axis_values,
        line_series,
        marker_x_values,
        marker_y_values,
        "Line",
        "Markers",
        "Series with markers",
    )

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert figure.data[0].mode == "lines"
    assert figure.data[0].name == "Line"
    assert figure.data[1].mode == "markers"
    assert figure.data[1].name == "Markers"
    assert list(figure.data[1].x) == marker_x_values.tolist()
    assert list(figure.data[1].y) == marker_y_values.tolist()


def test_plot_histogram_with_normal_curve_returns_histogram_and_fitted_curve():
    torch.manual_seed(0)
    sample_values = torch.randn(500)

    figure = plot_histogram_with_normal_curve(sample_values, number_of_bins=30, chart_title="Distribution")

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert isinstance(figure.data[0], go.Histogram)
    assert figure.data[0].nbinsx == 30
    assert isinstance(figure.data[1], go.Scatter)
    assert len(figure.data[1].x) == 200
    assert figure.layout.title.text == "Distribution"


def test_plot_sliding_window_highlights_input_and_output_windows():
    full_series = np.arange(50, dtype=float)
    input_window = torch.arange(10, 20, dtype=torch.float32)
    output_window = torch.arange(20, 25, dtype=torch.float32)

    figure = plot_sliding_window(
        full_series,
        input_window,
        output_window,
        window_start_index=10,
        chart_title="Sliding window",
    )

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 3

    full_series_trace, input_trace, output_trace = figure.data
    assert list(full_series_trace.y) == full_series.tolist()

    assert list(input_trace.x) == list(range(10, 20))
    assert list(input_trace.y) == input_window.tolist()

    assert list(output_trace.x) == list(range(20, 25))
    assert list(output_trace.y) == output_window.tolist()


def _build_sliding_window_dataloader(num_windows: int, input_window_size: int, batch_size: int) -> DataLoader:
    inputs = torch.arange(num_windows * input_window_size, dtype=torch.float32).reshape(
        num_windows, input_window_size
    )
    outputs = torch.zeros(num_windows, 1)
    return DataLoader(TensorDataset(inputs, outputs), batch_size=batch_size, shuffle=False)


def test_plot_full_sliding_progress_stops_at_max_windows_to_plot():
    full_series = np.arange(100, dtype=float)
    dataloader = _build_sliding_window_dataloader(num_windows=20, input_window_size=5, batch_size=4)

    figure = plot_full_sliding_progress(
        full_series,
        dataloader,
        input_window_size=5,
        max_windows_to_plot=7,
    )

    assert isinstance(figure, go.Figure)
    # first trace is the full series, followed by one trace per plotted window
    assert len(figure.data) == 1 + 7


def test_plot_full_sliding_progress_plots_all_windows_when_below_max():
    full_series = np.arange(100, dtype=float)
    dataloader = _build_sliding_window_dataloader(num_windows=20, input_window_size=5, batch_size=4)

    figure = plot_full_sliding_progress(
        full_series,
        dataloader,
        input_window_size=5,
        max_windows_to_plot=1000,
    )

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1 + 20
