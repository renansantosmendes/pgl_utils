"""
Deep Learning model architectures
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras


def draw_neural_network(model, max_nodes_per_layer=20):
    """
    Draw a visual representation of a neural network architecture.

    The graph size is proportional to the network size to ensure proper visualization
    of both small and large networks. For large networks, only a subset of nodes is
    displayed to maintain performance.

    Parameters
    ----------
    model : keras.Model
        A Keras/TensorFlow model to visualize
    max_nodes_per_layer : int, optional
        Maximum number of nodes to display per layer. Layers with more nodes will
        have nodes sampled. Default is 20.

    Returns
    -------
    None
        Displays the network architecture plot
    """
    G = nx.DiGraph()

    # Identificar as camadas e seus tamanhos
    layers = []
    # Camada de entrada
    input_shape = model.input_shape[1:]
    layers.append(("Input", input_shape[0]))

    # Camadas densas
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            layers.append((layer.name, layer.units))

    pos = {}
    node_colors = []

    # Calcular dimensões proporcionais
    n_layers = len(layers)
    max_nodes = max(n_nodes for _, n_nodes in layers)

    # Ajustar tamanho da figura proporcionalmente
    # Mínimo: 6x4, máximo: 16x12
    fig_width = max(6, min(16, n_layers * 1.5 + 3))
    # Usar max_nodes_per_layer para calcular altura
    fig_height = max(4, min(12, max_nodes_per_layer * 0.5 + 2))

    # Ajustar tamanho dos nós proporcionalmente (mínimo: 100, máximo: 400)
    node_size = max(100, min(400, 2000 / max_nodes_per_layer))

    # Ajustar tamanho da fonte proporcionalmente
    font_size = max(6, min(11, 100 / max_nodes_per_layer))
    arrow_size = max(8, min(15, 80 / max_nodes_per_layer))

    # Gerar nós e posições (com limite de nós por camada)
    nodes_per_layer = {}
    for i, (layer_name, n_nodes) in enumerate(layers):
        # Determinar quantos nós desenhar
        if n_nodes <= max_nodes_per_layer:
            nodes_to_draw = n_nodes
            shown_nodes = list(range(n_nodes))
        else:
            nodes_to_draw = max_nodes_per_layer
            # Amostrar nós distribuídos uniformemente
            step = n_nodes / max_nodes_per_layer
            shown_nodes = [int(j * step) for j in range(max_nodes_per_layer)]

        nodes_per_layer[i] = shown_nodes

        for idx, j in enumerate(shown_nodes):
            node_id = f"{i}_{j}"
            G.add_node(node_id)
            # Posicionamento: x = camada, y = neurônio (centralizado)
            pos[node_id] = (i, idx - nodes_to_draw / 2)
            node_colors.append(
                "lightgreen"
                if i == 0
                else ("orange" if i == len(layers) - 1 else "skyblue")
            )

    # Adicionar arestas entre camadas adjacentes (apenas entre nós desenhados)
    for i in range(len(layers) - 1):
        curr_layer_nodes = [f"{i}_{j}" for j in nodes_per_layer[i]]
        next_layer_nodes = [f"{i + 1}_{j}" for j in nodes_per_layer[i + 1]]
        # Amostrar arestas para evitar sobrecarga visual
        for idx_u, u in enumerate(curr_layer_nodes):
            for idx_v, v in enumerate(next_layer_nodes):
                G.add_edge(u, v)

    # Plotar
    plt.figure(figsize=(fig_width, fig_height))
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=node_size,
        node_color=node_colors,
        edge_color="gray",
        alpha=0.8,
        arrows=True,
        arrowsize=arrow_size,
        width=1.0,
    )

    # Adicionar legendas de camadas
    for i, (layer_name, n_nodes) in enumerate(layers):
        shown_count = len(nodes_per_layer[i])
        if shown_count < n_nodes:
            label = f"{layer_name}\n({shown_count}/{n_nodes})"
        else:
            label = f"{layer_name}\n({n_nodes})"

        plt.text(
            i,
            (shown_count / 2) + 1.2,
            label,
            horizontalalignment="center",
            fontsize=font_size,
            fontweight="bold",
        )

    # plt.title("Representação da Rede Neural", fontsize=int(font_size * 1.5))
    plt.axis("off")
    plt.tight_layout()
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
