from pgl_utils.mlops.experiments.base_tracker import BaseExperimentTracker

import os
import json
import time
import tempfile
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.keras
import mlflow.pytorch
import numpy as np


class MLflowTracker(BaseExperimentTracker):
    """
    Unified MLflow experiment tracker for scikit-learn, Keras and PyTorch.

    Handles run lifecycle, parameter/metric logging, artifact persistence,
    and framework-specific model registration in a single consistent API.
    Does NOT require PyTorch Lightning — works with any vanilla nn.Module
    training loop.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment. Created automatically if it does
        not exist.
    tracking_uri : str
        MLflow tracking server URI. Defaults to a local './mlruns' folder.
    tags : dict, optional
        Key-value tags attached to every run started by this tracker.

    Examples
    --------
    >>> with MLflowTracker("my_experiment") as tracker:
    ...     tracker.log_params({"lr": 1e-3, "epochs": 50})
    ...     tracker.log_metrics({"accuracy": 0.95})
    ...     tracker.log_sklearn_model(model, "random_forest")
    """

    _FLAVORS = {
        "sklearn": mlflow.sklearn,
        "keras": mlflow.keras,
        "pytorch": mlflow.pytorch,
    }

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "./mlruns",
        tags: dict | None = None,
    ):
        self.experiment_name = experiment_name
        self.tags = tags or {}
        self._run = None
        self._start_time: float = 0.0

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    # ──────────────────────────────────────────────────────────────
    #  Context manager
    # ──────────────────────────────────────────────────────────────

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)
        return False  # do not suppress exceptions

    # ──────────────────────────────────────────────────────────────
    #  Run lifecycle
    # ──────────────────────────────────────────────────────────────

    def start_run(self, run_name: str | None = None) -> "MLflowTracker":
        """
        Start a new MLflow run.

        Parameters
        ----------
        run_name : str, optional
            Human-readable label shown in the MLflow UI.

        Returns
        -------
        MLflowTracker
            Self, so the method can be chained.
        """
        self._run = mlflow.start_run(run_name=run_name, tags=self.tags)
        self._start_time = time.time()
        print(f"[MLflowTracker] Run started: {self._run.info.run_id}")
        return self

    def end_run(self, status: str = "FINISHED") -> None:
        """
        Finalize the active run and log its total wall-clock duration.

        Parameters
        ----------
        status : str
            MLflow terminal status. One of 'FINISHED', 'FAILED', 'KILLED'.
        """
        if self._run:
            elapsed = time.time() - self._start_time
            mlflow.log_metric("duration_seconds", round(elapsed, 2))
            mlflow.end_run(status=status)
            print(f"[MLflowTracker] Run ended ({status}) — {elapsed:.1f}s")
            self._run = None

    # ──────────────────────────────────────────────────────────────
    #  Parameters and metrics
    # ──────────────────────────────────────────────────────────────

    def log_params(self, params: dict) -> None:
        """
        Log a dictionary of hyperparameters.

        All values are coerced to strings, as required by MLflow.
        Call once per run; parameters are immutable after being logged.

        Parameters
        ----------
        params : dict
            Hyperparameter name-value pairs.
        """
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """
        Log a dictionary of scalar metrics, optionally at a given step.

        Can be called multiple times throughout training to record
        per-epoch or per-batch values.

        Parameters
        ----------
        metrics : dict
            Metric name-value pairs. Values must be numeric.
        step : int, optional
            Training step or epoch index associated with these values.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """
        Log a single scalar metric.

        Parameters
        ----------
        key : str
            Metric name.
        value : float
            Metric value.
        step : int, optional
            Training step or epoch index.
        """
        mlflow.log_metric(key, value, step=step)

    # ──────────────────────────────────────────────────────────────
    #  Generic artifacts
    # ──────────────────────────────────────────────────────────────

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """
        Upload an arbitrary local file as a run artifact.

        Parameters
        ----------
        local_path : str
            Path to the file on the local filesystem.
        artifact_path : str, optional
            Destination sub-directory inside the run's artifact store.
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict_as_json(self, data: dict, filename: str) -> None:
        """
        Serialize a dictionary to JSON and upload it as an artifact.

        Parameters
        ----------
        data : dict
            Data to serialize. Non-serializable values are coerced to str.
        filename : str
            Artifact destination path (e.g. 'config/params.json').
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2, default=str)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path=filename)
        os.unlink(tmp)

    def log_numpy_array(self, array: np.ndarray, filename: str) -> None:
        """
        Save a NumPy array to a .npy file and upload it as an artifact.

        Parameters
        ----------
        array : np.ndarray
            Array to persist.
        filename : str
            Artifact destination path (e.g. 'arrays/weights.npy').
        """
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, array)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path=filename)
        os.unlink(tmp)

    # ──────────────────────────────────────────────────────────────
    #  scikit-learn
    # ──────────────────────────────────────────────────────────────

    def log_sklearn_model(
        self,
        model,
        artifact_path: str = "sklearn_model",
        input_example: Any = None,
        register_as: str | None = None,
    ) -> None:
        """
        Log a fitted scikit-learn estimator and its metadata.

        Automatically extracts and logs:
        - ``model.get_params()`` as run parameters
        - ``feature_importances_`` as a .npy artifact (tree-based models)
        - ``classes_`` as a JSON artifact (classifiers)

        Parameters
        ----------
        model : sklearn estimator
            A fitted scikit-learn compatible model.
        artifact_path : str
            Artifact sub-directory for the serialized model.
        input_example : array-like, optional
            A sample input used to infer the model signature.
        register_as : str, optional
            If provided, registers the model in the MLflow Model Registry
            under this name.
        """
        self.log_params(model.get_params())

        if hasattr(model, "feature_importances_"):
            self.log_numpy_array(
                model.feature_importances_,
                filename="feature_importances.npy",
            )

        if hasattr(model, "classes_"):
            self.log_dict_as_json(
                {"classes": model.classes_.tolist()},
                filename="classes.json",
            )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            input_example=input_example,
            registered_model_name=register_as,
        )
        print(f"[MLflowTracker] sklearn model saved at '{artifact_path}'")

    def log_sklearn_cv_results(self, cv_results: dict) -> None:
        """
        Log the results of a GridSearchCV or RandomizedSearchCV object.

        Logs the best cross-validation score as a metric and the full
        result summary as a JSON artifact.

        Parameters
        ----------
        cv_results : dict
            A dict containing at least 'best_score_' and 'best_params_'
            keys, as returned by ``GridSearchCV.__dict__``.
        """
        summary = {
            "best_score": float(cv_results.get("best_score_", 0)),
            "best_params": cv_results.get("best_params_", {}),
        }
        self.log_metrics({"cv_best_score": summary["best_score"]})
        self.log_dict_as_json(summary, filename="cv_results.json")

    # ──────────────────────────────────────────────────────────────
    #  Keras
    # ──────────────────────────────────────────────────────────────

    def log_keras_model(
        self,
        model,
        artifact_path: str = "keras_model",
        input_example: Any = None,
        register_as: str | None = None,
    ) -> None:
        """
        Log a compiled Keras model and its architecture metadata.

        Automatically extracts and logs:
        - Trainable and non-trainable parameter counts as run params
        - Network architecture as a JSON artifact (via ``model.to_json()``)
        - Human-readable ``model.summary()`` as a text artifact

        Parameters
        ----------
        model : keras.Model
            A compiled (and optionally fitted) Keras model.
        artifact_path : str
            Artifact sub-directory for the serialized model.
        input_example : array-like, optional
            A sample input used to infer the model signature.
        register_as : str, optional
            If provided, registers the model in the MLflow Model Registry
            under this name.
        """
        trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
        self.log_params(
            {
                "trainable_params": trainable,
                "non_trainable_params": non_trainable,
                "total_params": trainable + non_trainable,
            }
        )

        try:
            self.log_dict_as_json(
                json.loads(model.to_json()),
                filename="architecture.json",
            )
        except Exception:
            pass

        summary_lines: list[str] = []
        model.summary(print_fn=lambda line: summary_lines.append(line))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(summary_lines))
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="model_summary.txt")
        os.unlink(tmp)

        mlflow.keras.log_model(
            model=model,
            artifact_path=artifact_path,
            input_example=input_example,
            registered_model_name=register_as,
        )
        print(f"[MLflowTracker] Keras model saved at '{artifact_path}'")

    def log_keras_history(self, history, prefix: str = "") -> None:
        """
        Log all metrics recorded in a Keras History object.

        Iterates over every epoch and calls ``log_metrics`` with the
        corresponding step index. Also persists the full history dict
        as a JSON artifact for offline analysis.

        Parameters
        ----------
        history : keras.callbacks.History
            The object returned by ``model.fit()``.
        prefix : str, optional
            String prepended to every metric key before logging.
        """
        hist_dict = history.history
        n_epochs = len(next(iter(hist_dict.values())))

        for epoch in range(n_epochs):
            self.log_metrics(
                {f"{prefix}{k}": float(v[epoch]) for k, v in hist_dict.items()},
                step=epoch + 1,
            )

        self.log_dict_as_json(
            {k: [float(x) for x in v] for k, v in hist_dict.items()},
            filename="training_history.json",
        )

    def keras_callback(self, log_every_n_epochs: int = 1):
        """
        Return a Keras Callback that logs metrics after each epoch.

        Pass the returned callback directly to ``model.fit()``.

        Parameters
        ----------
        log_every_n_epochs : int
            Log metrics every N epochs. Defaults to 1 (every epoch).

        Returns
        -------
        keras.callbacks.Callback
            A callback instance bound to this tracker's active run.

        Examples
        --------
        >>> model.fit(X, y, callbacks=[tracker.keras_callback()])
        """
        tracker = self

        try:
            import keras

            Callback = keras.callbacks.Callback
        except ImportError:
            from tensorflow import keras

            Callback = keras.callbacks.Callback

        class _MLflowCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % log_every_n_epochs == 0:
                    tracker.log_metrics(
                        {k: float(v) for k, v in (logs or {}).items()},
                        step=epoch + 1,
                    )

        return _MLflowCallback()

    # ──────────────────────────────────────────────────────────────
    #  PyTorch (vanilla nn.Module — no Lightning required)
    # ──────────────────────────────────────────────────────────────

    def log_pytorch_model(
        self,
        model,
        artifact_path: str = "pytorch_model",
        input_example: Any = None,
        register_as: str | None = None,
    ) -> None:
        """
        Log a PyTorch nn.Module and its architecture metadata.

        Works with any ``torch.nn.Module`` subclass trained with a
        plain PyTorch loop. Does NOT require PyTorch Lightning.

        Automatically extracts and logs:
        - Trainable and non-trainable parameter counts as run params
        - Full ``repr(model)`` architecture as a text artifact

        Parameters
        ----------
        model : torch.nn.Module
            A PyTorch model (fitted or not — weights are saved as-is).
        artifact_path : str
            Artifact sub-directory for the serialized model.
        input_example : torch.Tensor or array-like, optional
            A sample input used to infer the model signature.
        register_as : str, optional
            If provided, registers the model in the MLflow Model Registry
            under this name.
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
        self.log_params(
            {
                "trainable_params": trainable,
                "non_trainable_params": non_trainable,
                "total_params": trainable + non_trainable,
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(repr(model))
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="model_architecture.txt")
        os.unlink(tmp)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            input_example=input_example,
            registered_model_name=register_as,
        )
        print(f"[MLflowTracker] PyTorch model saved at '{artifact_path}'")

    def log_pytorch_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        extra: dict | None = None,
        artifact_path: str = "checkpoints",
    ) -> None:
        """
        Save a full PyTorch training checkpoint as a run artifact.

        The checkpoint includes model weights, optimizer state, current
        epoch, and any extra data provided (e.g. scheduler state, loss).
        Useful for resuming training or post-hoc analysis.

        Parameters
        ----------
        model : torch.nn.Module
            The model whose ``state_dict`` will be saved.
        optimizer : torch.optim.Optimizer
            The optimizer whose ``state_dict`` will be saved.
        epoch : int
            Current training epoch (used to label the checkpoint).
        extra : dict, optional
            Additional key-value pairs merged into the checkpoint dict
            (e.g. ``{"loss": 0.34, "scheduler": sched.state_dict()}``).
        artifact_path : str
            Artifact sub-directory for the checkpoint file.
        """
        import torch

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            **(extra or {}),
        }
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(payload, f.name)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path=artifact_path)
        os.unlink(tmp)
        print(f"[MLflowTracker] Checkpoint epoch {epoch} saved at '{artifact_path}'")

    def pytorch_epoch_log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        extra_metrics: dict | None = None,
    ) -> None:
        """
        Convenience wrapper to log per-epoch metrics in a PyTorch loop.

        Logs ``train_loss`` and, optionally, ``val_loss`` plus any
        additional metrics, all keyed to the given epoch step.

        Parameters
        ----------
        epoch : int
            Current epoch index (1-based recommended for readability).
        train_loss : float
            Training loss for this epoch.
        val_loss : float, optional
            Validation loss for this epoch.
        extra_metrics : dict, optional
            Additional scalar metrics to log at this step
            (e.g. ``{"accuracy": 0.93, "lr": 1e-4}``).

        Examples
        --------
        >>> for epoch in range(1, 51):
        ...     loss = train_one_epoch(model, loader, optimizer)
        ...     tracker.pytorch_epoch_log(epoch, train_loss=loss)
        """
        metrics = {"train_loss": train_loss}
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if extra_metrics:
            metrics.update(extra_metrics)
        self.log_metrics(metrics, step=epoch)

    # ──────────────────────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────────────────────

    def log_environment(self) -> None:
        """
        Log the versions of key libraries present in the current environment.

        Captures Python, NumPy, scikit-learn, TensorFlow, Keras, PyTorch,
        Pandas and MLflow versions. Saves them as a JSON artifact and also
        logs them as run parameters prefixed with 'env_'.
        """
        import sys

        versions: dict = {"python": sys.version}
        for lib in [
            "numpy",
            "sklearn",
            "tensorflow",
            "keras",
            "torch",
            "mlflow",
            "pandas",
        ]:
            try:
                versions[lib] = __import__(lib).__version__
            except ImportError:
                pass
        self.log_dict_as_json(versions, filename="environment.json")
        self.log_params({f"env_{k}": v.split()[0] for k, v in versions.items()})

    @property
    def run_id(self) -> str | None:
        """Active run ID, or None if no run is in progress."""
        return self._run.info.run_id if self._run else None

    @property
    def run_url(self) -> str | None:
        """
        Best-effort URL to view this run in the MLflow UI.

        Returns None if no run is active.
        """
        if not self._run:
            return None
        uri = mlflow.get_tracking_uri()
        return f"{uri}/#/experiments/.../runs/{self.run_id}"


# ══════════════════════════════════════════════════════════════════
#  Usage examples
# ══════════════════════════════════════════════════════════════════


def example_sklearn():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with MLflowTracker("iris_sklearn") as tracker:
        tracker.log_environment()
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        tracker.log_metrics(
            {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
            }
        )
        tracker.log_sklearn_model(model, register_as="IrisClassifier")
        print("sklearn run_id:", tracker.run_id)


def example_keras():
    import tensorflow as tf

    X = np.random.rand(500, 20).astype("float32")
    y = (X[:, 0] > 0.5).astype("float32")

    with MLflowTracker("binary_keras") as tracker:
        tracker.log_environment()
        tracker.log_params(
            {
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "epochs": 10,
                "batch_size": 32,
            }
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu", input_shape=(20,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        history = model.fit(
            X,
            y,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[tracker.keras_callback()],
        )
        tracker.log_keras_history(history)
        tracker.log_keras_model(model, register_as="BinaryKerasModel")
        print("keras run_id:", tracker.run_id)


def example_pytorch():
    """Plain PyTorch training loop — no Lightning, no extra abstractions."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.rand(400, 16)
    y = (X[:, 0] > 0.5).float().unsqueeze(1)
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    with MLflowTracker("binary_pytorch") as tracker:
        tracker.log_environment()
        tracker.log_params(
            {
                "optimizer": "Adam",
                "lr": 1e-3,
                "epochs": 10,
                "batch_size": 32,
            }
        )

        for epoch in range(1, 11):
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            model.eval()
            with torch.no_grad():
                preds = (model(X) > 0.5).float()
                acc = (preds == y).float().mean().item()

            tracker.pytorch_epoch_log(
                epoch,
                train_loss=avg_loss,
                extra_metrics={"accuracy": acc},
            )

            if epoch % 5 == 0:
                tracker.log_pytorch_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    extra={"train_loss": avg_loss},
                )

        tracker.log_pytorch_model(
            model,
            input_example=X[:1],
            register_as="BinaryPyTorchModel",
        )
        print("pytorch run_id:", tracker.run_id)


if __name__ == "__main__":
    print("=== scikit-learn ===")
    example_sklearn()
    print("\n=== Keras ===")
    example_keras()
    print("\n=== PyTorch ===")
    example_pytorch()
    print("\nOpen the UI:  mlflow ui --backend-store-uri ./mlruns")
