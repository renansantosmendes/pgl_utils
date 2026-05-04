"""
Deep Learning model architectures
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras


def draw_neural_network(
        model, 
        max_nodes_per_layer=20, 
        show_input_layer=True,
        show_weights=False,
        figsize=None,
        ) -> None:
    """
    Draw a visual representation of a neural network architecture.

    Parameters
    ----------
    model : keras.Model
        A Keras/TensorFlow model to visualize
    max_nodes_per_layer : int, optional
        Maximum number of nodes to display per layer. Default is 20.
    show_input_layer : bool, optional
        If False, the input layer will not be displayed.
    show_weights : bool, optional
        If True, edge thickness and color reflect the model's actual weights.
        Positive weights are shown in blue, negative in red, and near-zero
        weights become nearly invisible. Default is False.
    figsize : tuple of (float, float), optional
        Figure size as (width, height) in inches. If None, the size is
        calculated automatically based on the network architecture.
        Example: figsize=(8, 4)

    Returns
    -------
    None
        Displays the network architecture plot
    """
    import numpy as np

    G = nx.DiGraph()

    layers = []
    input_shape = model.input_shape[1:]
    layers.append(("Input", input_shape[0]))

    dense_layers = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            layers.append((layer.name, layer.units))
            dense_layers.append(layer)

    if not show_input_layer:
        layers = layers[1:]

    pos = {}
    node_colors = []

    n_layers = len(layers)
    max_nodes_shown = min(max_nodes_per_layer, max(n for _, n in layers))

    h_spacing = max(3.0, min(5.0, 20.0 / n_layers))
    v_spacing = max(1.0, min(2.0, 15.0 / max_nodes_shown))

    if figsize is not None:
        fig_width, fig_height = figsize
    else:
        fig_width = max(8, min(18, n_layers * h_spacing * 0.6 + 2))
        fig_height = max(5, min(14, max_nodes_shown * v_spacing * 0.55 + 2))

    node_size = max(200, min(800, 5000 / max_nodes_shown))
    font_size = max(7, min(12, 150 / max_nodes_shown))
    arrow_size = max(5, min(12, 60 / max_nodes_shown))
    base_edge_width = max(0.3, min(1.0, 8.0 / max_nodes_shown))
    base_edge_alpha = max(0.15, min(0.5, 5.0 / max_nodes_shown))
    node_alpha = 0.75

    nodes_per_layer = {}
    for i, (layer_name, n_nodes) in enumerate(layers):
        if n_nodes <= max_nodes_per_layer:
            shown_nodes = list(range(n_nodes))
        else:
            step = n_nodes / max_nodes_per_layer
            shown_nodes = [int(j * step) for j in range(max_nodes_per_layer)]

        nodes_per_layer[i] = shown_nodes
        n_shown = len(shown_nodes)

        for idx, j in enumerate(shown_nodes):
            node_id = f"{i}_{j}"
            G.add_node(node_id)
            y = (idx - (n_shown - 1) / 2.0) * v_spacing
            x = i * h_spacing
            pos[node_id] = (x, y)

            if i == 0 and show_input_layer:
                node_colors.append("lightgreen")
            elif i == len(layers) - 1:
                node_colors.append("orange")
            else:
                node_colors.append("skyblue")

    edge_list = []
    edge_weights_vals = []

    for i in range(len(layers) - 1):
        curr_nodes = nodes_per_layer[i]
        next_nodes = nodes_per_layer[i + 1]

        if show_weights and i < len(dense_layers):
            W = dense_layers[i].get_weights()[0]
            for u_node in curr_nodes:
                for v_node in next_nodes:
                    edge = (f"{i}_{u_node}", f"{i+1}_{v_node}")
                    G.add_edge(*edge)
                    edge_list.append(edge)
                    if u_node < W.shape[0] and v_node < W.shape[1]:
                        edge_weights_vals.append(W[u_node, v_node])
                    else:
                        edge_weights_vals.append(0.0)
        else:
            for u_node in curr_nodes:
                for v_node in next_nodes:
                    edge = (f"{i}_{u_node}", f"{i+1}_{v_node}")
                    G.add_edge(*edge)
                    edge_list.append(edge)
                    edge_weights_vals.append(0.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if show_weights and len(edge_weights_vals) > 0:
        weights_arr = np.array(edge_weights_vals)
        abs_weights = np.abs(weights_arr)
        w_max = abs_weights.max() if abs_weights.max() > 0 else 1.0
        norm_weights = abs_weights / w_max

        max_width = max(2.5, min(4.0, 20.0 / max_nodes_shown))
        min_width = 0.1
        widths = min_width + norm_weights * (max_width - min_width)

        min_alpha = 0.03
        max_alpha = 0.85
        colors = []
        for w, nw in zip(weights_arr, norm_weights):
            a = min_alpha + nw * (max_alpha - min_alpha)
            if w >= 0:
                colors.append((0.1, 0.3, 0.9, a))
            else:
                colors.append((0.9, 0.15, 0.15, a))

        for idx, (u, v) in enumerate(edge_list):
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=[(u, v)],
                edge_color=[colors[idx]],
                width=widths[idx],
                arrows=True,
                arrowsize=arrow_size,
                node_size=node_size,
            )
    else:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edge_list,
            edge_color="gray",
            width=base_edge_width,
            alpha=base_edge_alpha,
            arrows=True,
            arrowsize=arrow_size,
            node_size=node_size,
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=node_colors,
        edgecolors="gray",
        linewidths=0.5,
        alpha=node_alpha,
    )

    for i, (layer_name, n_nodes) in enumerate(layers):
        n_shown = len(nodes_per_layer[i])
        top_y = ((n_shown - 1) / 2.0) * v_spacing
        label_y = top_y + v_spacing * 0.8

        if n_shown < n_nodes:
            label = f"{layer_name}\n({n_shown}/{n_nodes} neurons)"
        else:
            label = f"{layer_name}\n({n_nodes} neurons)"

        ax.text(
            i * h_spacing,
            label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=font_size,
            fontweight="bold",
            color="dimgray",
        )

    if show_weights:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=(0.1, 0.3, 0.9), linewidth=2.5, label="Peso positivo"),
            Line2D([0], [0], color=(0.9, 0.15, 0.15), linewidth=2.5, label="Peso negativo"),
            Line2D([0], [0], color="gray", linewidth=0.5, alpha=0.3, label="Peso ≈ 0"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=font_size,
            framealpha=0.8,
        )

    ax.axis("off")
    ax.margins(x=0.08, y=0.12)
    fig.tight_layout()
    plt.show()


def placeholder():
    """
    Placeholder function
    """
    return "DL architectures utilities"


def plot_convergence(history):
    """
    Plot the convergence of training and validation loss over epochs.

    This function visualizes how the model's loss changes during training,
    helping to identify overfitting or underfitting patterns.

    Parameters
    ----------
    history : keras.callbacks.History
        Training history object returned by model.fit() containing loss and val_loss
        metrics from each epoch

    Returns
    -------
    None
        Displays the convergence plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title("Convergência da Função de Perda (Loss)")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_all_weights(model):
    """
    Extract all weights from all layers of a neural network model.

    Collects weights from each layer that has trainable parameters and returns
    them as a single flattened array. Biases are excluded.

    Parameters
    ----------
    model : keras.Model
        A Keras/TensorFlow model to extract weights from

    Returns
    -------
    numpy.ndarray
        1D array containing all flattened weights from the model's layers

    Notes
    -----
    - Only weight matrices are included (biases are excluded)
    - Layers without weights are skipped
    - All weights are concatenated into a single 1D array
    """
    weights_list = []
    for layer in model.layers:
        if hasattr(layer, "get_weights") and len(layer.get_weights()) > 0:
            # Pegamos apenas os pesos (índice 0), ignorando bias (índice 1)
            weights_list.append(layer.get_weights()[0].flatten())
    return np.concatenate(weights_list)


def plot_weight_distribution(weights, title):
    """
    Plot the distribution of neural network weights as a histogram.

    Visualizes the distribution of weight values from a model, which can help
    identify issues like weight saturation or dying neurons.

    Parameters
    ----------
    weights : numpy.ndarray or array-like
        1D array of weight values to plot
    title : str
        Title for the histogram plot

    Returns
    -------
    None
        Displays the weight distribution histogram

    Notes
    -----
    - Uses kernel density estimation (KDE) to show smooth distribution
    - Helps diagnose weight initialization and training issues
    - Can be used with output from get_all_weights()
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(weights, kde=True, color="skyblue")
    plt.title(title)
    plt.xlabel("Valor do Peso")
    plt.ylabel("Frequência")
    plt.grid(True, alpha=0.3)
    plt.show()

def identify_outliers(scores, percentile=95):
    """
    Identify outlier windows based on a percentile threshold.

    Args:
        scores: array of anomaly scores (one per window)
        percentile: threshold percentile (e.g., 95 means top 5% are outliers)

    Returns:
        threshold: the computed threshold value
        outlier_mask: boolean array (True = outlier)
        outlier_indices: integer indices of outlier windows
    """
    threshold = np.percentile(scores, percentile)
    outlier_mask = scores > threshold
    outlier_indices = np.where(outlier_mask)[0]
    return threshold, outlier_mask, outlier_indices


def build_outlier_report(outlier_indices, reconstruction_error, kl_per_sample,
                         combined_score, close_prices, window_size):
    """
    Build a detailed DataFrame with information about each outlier window.

    For each outlier, shows the date range, anomaly scores, and price change
    during that window — enabling correlation with real market events.
    """
    records = []
    dates = close_prices.index

    for idx in outlier_indices:
        # A janela idx corresponde aos retornos de (idx+1) a (idx+window_size)
        start_date = dates[idx + 1]
        end_date = dates[idx + window_size]
        start_price = close_prices.iloc[idx + 1]
        end_price = close_prices.iloc[idx + window_size]
        price_change_pct = ((end_price - start_price) / start_price) * 100

        records.append({
            'window_idx': idx,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'recon_error': round(reconstruction_error[idx], 4),
            'kl_divergence': round(kl_per_sample[idx], 4),
            'combined_score': round(combined_score[idx], 4),
            'price_change_%': round(price_change_pct, 2),
        })

    df = pd.DataFrame(records)
    df = df.sort_values('combined_score', ascending=False).reset_index(drop=True)
    return df

def plot_top_outlier_windows(model, windows_array, outlier_indices,
                              combined_score, close_prices, n_top=6):
    """
    Plot the top-N most anomalous windows alongside their VAE reconstruction.

    Each subplot shows the original return window (blue) vs the reconstructed
    window (red dashed). Large gaps between the two indicate the specific
    days within the window that the model found most unusual.
    """
    # Sort by combined score descending
    sorted_idx = outlier_indices[np.argsort(combined_score[outlier_indices])[::-1]]
    top_idx = sorted_idx[:n_top]

    reconstructed, _, _ = model(windows_array, training=False)
    reconstructed = reconstructed.numpy()

    cols = 3
    rows = (n_top + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle(f'Top {n_top} janelas mais anômalas — Original vs Reconstrução',
                 fontsize=13, fontweight='bold')
    axes = axes.flatten()

    dates = close_prices.index

    for i, w_idx in enumerate(top_idx):
        ax = axes[i]
        original = windows_array[w_idx]
        recon = reconstructed[w_idx]

        ax.plot(original, linewidth=1.2, color='steelblue', label='Original')
        ax.plot(recon, linewidth=1.2, color='firebrick', linestyle='--', label='Reconstruído')
        ax.fill_between(range(len(original)), original, recon,
                        alpha=0.15, color='firebrick')

        start_date = dates[w_idx + 1].strftime('%Y-%m-%d')
        end_date = dates[w_idx + WINDOW_SIZE].strftime('%Y-%m-%d')
        score = combined_score[w_idx]

        ax.set_title(f'{start_date} → {end_date}\nscore: {score:.2f}', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlabel('Dia na janela', fontsize=8)
        ax.set_ylabel('Retorno (norm.)', fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    if 'MLFLOW_RUN_ID' in globals():
        with mlflow.start_run(run_id=MLFLOW_RUN_ID):
            mlflow.log_figure(fig, 'plots/top_outlier_windows.png')
            
def plot_outlier_detection(
    close_prices,
    reconstruction_error,
    kl_per_sample,
    threshold_recon,
    idx_recon,
    idx_comb,
    mask_comb,
    ticker,
    window_size,
    percentile_threshold,
    mlflow_run_id=None,
):
    """Plot a 4-panel outlier detection dashboard for a VAE-based anomaly detector.

    Panels:
        [0,0] Reconstruction error time series with threshold line and flagged outliers.
        [0,1] Price history with outlier windows overlaid.
        [1,0] Histogram of reconstruction errors with threshold.
        [1,1] Scatter of reconstruction error vs KL divergence.

    Args:
        close_prices: pd.Series of closing prices (DatetimeIndex).
        reconstruction_error: 1-D array of per-window MAE values.
        kl_per_sample: 1-D array of per-window KL divergence values.
        threshold_recon: scalar threshold for the reconstruction error.
        idx_recon: integer indices of outlier windows (reconstruction criterion).
        idx_comb: integer indices of outlier windows (combined criterion).
        mask_comb: boolean array (True = outlier by combined criterion).
        ticker: string with the asset ticker symbol.
        window_size: int, size of the sliding window used to build samples.
        percentile_threshold: int, percentile used to define the threshold.
        mlflow_run_id: optional MLflow run ID to log the figure as an artifact.

    Returns:
        The matplotlib Figure object.
    """
    dates = close_prices.index[window_size + 1:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Detecção de Outliers — {ticker} (VAE)', fontsize=14, fontweight='bold')

    # ── Painel 1: Série temporal do erro de reconstrução ──
    ax = axes[0, 0]
    ax.plot(dates, reconstruction_error, linewidth=0.6, alpha=0.8, color='steelblue')
    ax.axhline(threshold_recon, color='firebrick', linestyle='--', linewidth=1,
               label=f'Threshold (p{percentile_threshold}): {threshold_recon:.4f}')
    ax.scatter(dates[idx_recon], reconstruction_error[idx_recon],
               color='firebrick', s=20, zorder=5, label=f'Outliers ({len(idx_recon)})')
    ax.set_title('Erro de reconstrução ao longo do tempo')
    ax.set_ylabel('MAE')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Painel 2: Preço com outliers sobrepostos ──
    ax = axes[0, 1]
    ax.plot(close_prices.index, close_prices.values, linewidth=0.8, color='steelblue', label='Preço')
    outlier_dates = dates[idx_comb]
    outlier_prices = close_prices.reindex(outlier_dates)
    ax.scatter(outlier_prices.index, outlier_prices.values,
               color='firebrick', s=25, zorder=5, label=f'Outliers ({len(idx_comb)})')
    ax.set_title(f'Preço de {ticker} com outliers marcados')
    ax.set_ylabel('Preço (USD)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Painel 3: Distribuição dos scores ──
    ax = axes[1, 0]
    ax.hist(reconstruction_error, bins=50, alpha=0.6, color='steelblue',
            edgecolor='white', label='Reconstrução')
    ax.axvline(threshold_recon, color='firebrick', linestyle='--', linewidth=1.5,
               label=f'Threshold (p{percentile_threshold})')
    ax.set_title('Distribuição do erro de reconstrução')
    ax.set_xlabel('MAE')
    ax.set_ylabel('Frequência')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Painel 4: Scatter reconstrução × KL ──
    ax = axes[1, 1]
    normal_mask = ~mask_comb
    ax.scatter(reconstruction_error[normal_mask], kl_per_sample[normal_mask],
               alpha=0.3, s=8, color='steelblue', label='Normal')
    ax.scatter(reconstruction_error[mask_comb], kl_per_sample[mask_comb],
               alpha=0.8, s=25, color='firebrick', label='Outlier',
               edgecolors='darkred', linewidths=0.5)
    ax.set_title('Reconstrução × KL divergence')
    ax.set_xlabel('Erro de reconstrução (MAE)')
    ax.set_ylabel('KL divergence')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    if mlflow_run_id is not None:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_figure(fig, 'plots/outlier_detection.png')

    return fig