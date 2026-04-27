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
