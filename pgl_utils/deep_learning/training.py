import os


def set_reprodutibility(seed: int = DEFAULT_SEED) -> None:
    """
    Configure global determinism for reproducibility across Python, NumPy,
    and backend engine.

    Args:
        seed (int): Seed value used to control randomness.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)

    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        print("Determinismo configurado com sucesso.")
    except RuntimeError as e:
        print(f"Erro ao configurar threads: {e}")
        print(
            "DICA: Reinicie o kernel e execute esta função como a primeira operação do TensorFlow."
        )
