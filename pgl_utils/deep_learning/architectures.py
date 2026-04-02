"""
Deep Learning model architectures
"""

import networkx as nx
import matplotlib.pyplot as plt
from tensorflow import keras


def draw_neural_network(model):
    """
    Draw a visual representation of a neural network architecture.

    Parameters
    ----------
    model : keras.Model
        A Keras/TensorFlow model to visualize

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

    # Gerar nós e posições
    for i, (layer_name, n_nodes) in enumerate(layers):
        nodes_to_draw = n_nodes
        for j in range(nodes_to_draw):
            node_id = f"{i}_{j}"
            G.add_node(node_id)
            # Posicionamento: x = camada, y = neurônio (centralizado)
            pos[node_id] = (i, j - nodes_to_draw / 2)
            node_colors.append(
                "lightgreen"
                if i == 0
                else ("orange" if i == len(layers) - 1 else "skyblue")
            )

    # Adicionar arestas entre camadas adjacentes
    for i in range(len(layers) - 1):
        curr_layer_nodes = [n for n in G.nodes if n.startswith(f"{i}_")]
        next_layer_nodes = [n for n in G.nodes if n.startswith(f"{i + 1}_")]
        for u in curr_layer_nodes:
            for v in next_layer_nodes:
                G.add_edge(u, v)

    # Plotar
    plt.figure(figsize=(14, 10))
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=300,
        node_color=node_colors,
        edge_color="gray",
        alpha=0.3,
        arrows=True,
        arrowsize=10,
    )

    # Adicionar legendas de camadas
    for i, (layer_name, n_nodes) in enumerate(layers):
        plt.text(
            i,
            (n_nodes / 2) + 0.8,
            f"{layer_name}\n({n_nodes} neurônios)",
            horizontalalignment="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.title("Representação Completa da Rede Neural", fontsize=15)
    plt.axis("off")
    plt.show()


def placeholder():
    """
    Placeholder function
    """
    return "DL architectures utilities"
