"""
PyTorch + MLflowTracker — binary classification example.

Demonstrates every tracker feature available for PyTorch:
  - log_environment        : library versions as params + artifact
  - log_params             : hyperparameters
  - pytorch_epoch_log      : train_loss, val_loss, accuracy per epoch
  - log_pytorch_checkpoint : full checkpoint (model + optimizer) every 5 epochs
  - log_pytorch_model      : final model weights + architecture artifact + registry
  - log_image              : loss curve saved as PNG artifact

DagHub remote storage (optional) — set env vars before running:
  DAGSHUB_URL   = https://dagshub.com/<user>/<repo>
  DAGSHUB_USER  = <username>
  DAGSHUB_TOKEN = <token>
"""

import sys
import os
import tempfile
os.environ['TF_CPP_MAX_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

from pgl_utils.mlops.experiments.mlflow_tracker import MLflowTracker

from dotenv import load_dotenv

load_dotenv()

torch.manual_seed(42)

HPARAMS = {
    "input_dim": 20,
    "hidden_dim": 64,
    "optimizer": "Adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 20,
    "batch_size": 32,
    "val_split": 0.2,
    "checkpoint_every": 5,
}


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dataset(n: int = 600, input_dim: int = 20) -> TensorDataset:
    X = torch.rand(n, input_dim)
    # label = 1 when sum of first 5 features > 2.5
    y = (X[:, :5].sum(dim=1) > 2.5).float().unsqueeze(1)
    return TensorDataset(X, y)


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return ((preds > 0.5) == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(out.detach(), yb)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for xb, yb in loader:
        out = model(xb)
        total_loss += criterion(out, yb).item()
        total_acc += accuracy(out, yb)
    n = len(loader)
    return total_loss / n, total_acc / n


def save_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train_loss")
    ax.plot(epochs, val_losses, label="val_loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    dataset = make_dataset(n=800, input_dim=HPARAMS["input_dim"])

    val_size = int(len(dataset) * HPARAMS["val_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=HPARAMS["batch_size"])

    model = BinaryMLP(HPARAMS["input_dim"], HPARAMS["hidden_dim"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"],
    )
    criterion = nn.BCELoss()

    sample_input = torch.rand(1, HPARAMS["input_dim"])


    dagshub_url = os.getenv("DAGSHUB_URL")
    dagshub_user = os.getenv("DAGSHUB_USER")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    
    with MLflowTracker(
        experiment_name="pytorch_binary_classification",
        run_name="MLflowTracker + PyTorch Example 2",
        tags={"framework": "pytorch", "task": "binary_classification"},
        dagshub_url=dagshub_url,
        dagshub_user=dagshub_user,
        dagshub_token=dagshub_token,
    ) as tracker:

        tracker.log_environment()
        tracker.log_params(HPARAMS)

        train_losses, val_losses = [], []

        for epoch in range(1, HPARAMS["epochs"] + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            tracker.pytorch_epoch_log(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                extra_metrics={
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                },
            )

            print(
                f"Epoch {epoch:02d}/{HPARAMS['epochs']} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if epoch % HPARAMS["checkpoint_every"] == 0:
                tracker.log_pytorch_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    extra={"train_loss": train_loss, "val_loss": val_loss},
                )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            curve_path = f.name
        save_loss_curve(train_losses, val_losses, curve_path)
        tracker.log_image(curve_path, artifact_path="plots/loss_curve.png")
        os.unlink(curve_path)

        tracker.log_pytorch_model(
            model,
            artifact_path="pytorch_model",
            input_example=sample_input,
            register_as="BinaryMLPClassifier",
        )

        print(f"\nRun ID : {tracker.run_id}")
        print("To view results run:  mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
