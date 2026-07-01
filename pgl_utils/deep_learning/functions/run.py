# -*- coding: utf-8 -*-
"""
Casting Manufacturing Defect Detection Training Script

This script trains a CNN model for detecting defects in casting manufacturing products.
Uses MLflow for experiment tracking and includes comprehensive visualization tools.
"""

# Note: Install mlflow with: pip install mlflow

import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F

# Import from casting_manufacturing_product module
from pgl_utils.deep_learning.problems.casting_manufacturing_product import (
    CropCircular,
    CutOut,
    show_examples,
    show_normalized,
    grad_cam,
    show_grad_cam,
    show_predictions_with_gradcam,
    inspect_batch,
    plot_kernels,
    plot_feature_maps,
    plot_convergence_curves,
    plot_confusion_matrix,
)

# -- MLflow tracking ---------------------------------------------------------
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/renansantosmendes/industrial_application.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'renansantosmendes'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'e3322db74a3dab223aed50dd34da7144b00b58f4'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment('casting-defect-cnn')

# ── Dataset ───────────────────────────────────────────────────────────────────
path = kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")

base_dir = None
for root, dirs, files in os.walk(path):
    if "train" in dirs and "test" in dirs:
        base_dir = root
        break

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# ── Hiperparâmetros ───────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Wrapper para aplicar transforms diferentes por split ─────────────────────
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

# ── CutOut augmentation and CropCircular ─────────────────────────────────────
# Note: CropCircular and CutOut are now imported from casting_manufacturing_product module

# -- NormalizaPorImagem: remove brilho absoluto, forca aprendizado de textura --
class NormalizaPorImagem:
    def __call__(self, img):
        mean = img.mean()
        std = img.std() + 1e-8
        return (img - mean) / std

# -- Transforms --------------------------------------------------------------
# CropCircular aplicado antes de tudo: peca centralizada, fundo removido
base_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    CropCircular(margin=0.08),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])

aug_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.92, 1.0)),
    transforms.ToTensor(),
    NormalizaPorImagem(),
    CutOut(n_holes=1, length=20),
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    NormalizaPorImagem(),
])

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    CropCircular(margin=0.08),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NormalizaPorImagem(),
])
# ── Datasets e DataLoaders ────────────────────────────────────────────────────
full_base = datasets.ImageFolder(train_dir, transform=base_transforms)
test_ds = datasets.ImageFolder(test_dir, transform=test_transforms)

val_size = int(len(full_base) * VAL_SPLIT)
train_size = len(full_base) - val_size

indices = torch.randperm(len(full_base), generator=torch.Generator().manual_seed(42)).tolist()
train_idx = indices[val_size:]
val_idx = indices[:val_size]

raw_train = torch.utils.data.Subset(full_base, train_idx)
raw_val = torch.utils.data.Subset(full_base, val_idx)

train_ds = TransformSubset(raw_train, aug_transforms)
val_ds = TransformSubset(raw_val, val_transforms)

_pin = DEVICE.type == 'cuda'
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=_pin, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=_pin, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=_pin, persistent_workers=True)

CLASS_NAMES = full_base.classes
print(f"Classes: {CLASS_NAMES}")
print(f"Treino: {train_size} | Validação: {val_size} | Teste: {len(test_ds)}")

# ── Visualizar exemplos ───────────────────────────────────────────────────────
# Note: show_examples is now imported from casting_manufacturing_product module

show_examples(train_loader, CLASS_NAMES)

# -- Visualizar imagens normalizadas por classe ------------------------------
# Note: show_normalized is now imported from casting_manufacturing_product module

show_normalized(train_loader, CLASS_NAMES, n=6)

show_normalized(train_loader, CLASS_NAMES, n=6)

# ── Arquitetura CNN ───────────────────────────────────────────────────────────
class CastingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Bloco 1 - 224 -> 112
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloco 2 - 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloco 3 - sem pooling, mantem 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Bloco 4 - sem pooling, mantem 56x56
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Bloco 5 - sem pooling, mantem 56x56
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),    # 56x56 -> 7x7, preserva estrutura espacial
            nn.Flatten(),              # 256 x 7 x 7 = 12544
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CastingCNN().to(DEVICE)
if DEVICE.type == 'cuda':
    model = torch.compile(model)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parametros treinaveis: {total_params:,}")

# Mixed precision (AMP) -- ~2x mais rapido na GPU T4
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')

# ── Treinamento ────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

LOSS_SPIKE_THRESH = 2.0
INSPECT_N = 32

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
spike_log = []

# Note: inspect_batch is now imported from casting_manufacturing_product module
# We need a wrapper to match the local signature
def inspect_batch_local(images, labels, losses, outputs, epoch, split, batch_idx):
    inspect_batch(images, labels, losses, outputs, epoch, split, batch_idx,
                  model, DEVICE, IMG_SIZE, CLASS_NAMES, inspect_n=INSPECT_N)

def run_epoch(loader, training=True, epoch=0, split="train"):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = training and DEVICE.type == 'cuda'
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                outputs = model(images)
                losses_each = criterion(outputs, labels)
                loss = losses_each.mean()
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += losses_each.detach().float().sum().item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    epoch_mean = total_loss / total
    ref_losses = history[split + "_loss"]
    recent_mean = np.mean(ref_losses[-3:]) if len(ref_losses) >= 3 else epoch_mean
    spike_threshold = max(recent_mean * LOSS_SPIKE_THRESH, 0.5)

    if epoch_mean > spike_threshold and len(ref_losses) >= 1:
        spike_log.append((epoch, split, epoch_mean))
        print("  >> Spike detectado — " + split + " epoch " + str(epoch) +
              ": loss=" + f"{epoch_mean:.4f}  threshold={spike_threshold:.4f}")
        # coleta losses de todos os batches e mostra os piores
        model.eval()
        all_imgs, all_lbls, all_outs, all_losses = [], [], [], []
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                outputs_batch = model(images)
                losses_each = criterion(outputs_batch, labels)
            all_imgs.append(images.cpu())
            all_lbls.append(labels.cpu())
            all_outs.append(outputs_batch.detach().cpu())
            all_losses.append(losses_each.detach().cpu())
        all_imgs = torch.cat(all_imgs)
        all_lbls = torch.cat(all_lbls)
        all_outs = torch.cat(all_outs)
        all_losses = torch.cat(all_losses)
        # passa os piores INSPECT_N exemplos para inspect_batch
        inspect_batch_local(all_imgs, all_lbls, all_losses,
                             all_outs, epoch, split, batch_idx=-1)
        if training:
            model.train()

    return epoch_mean, correct / total

with mlflow.start_run():
    # -- log hiperparametros --
    mlflow.log_params({
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'lr': LR,
        'val_split': VAL_SPLIT,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'amp': str(DEVICE.type == 'cuda'),
    })

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, training=True, epoch=epoch, split='train')
        vl_loss, vl_acc = run_epoch(val_loader, training=False, epoch=epoch, split='val')
        scheduler.step(vl_loss)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        print(f'Epoch {epoch:02d}/{EPOCHS}  '
              f'loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  '
              f'val_loss: {vl_loss:.4f}  val_acc: {vl_acc:.4f}')
        mlflow.log_metrics({
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'val_loss': vl_loss,
            'val_acc': vl_acc,
        }, step=epoch)
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), 'best_model.pth')
            mlflow.log_artifact('best_model.pth')

    if spike_log:
        print('\n-- Resumo dos spikes --')
        for ep, sp, val in spike_log:
            print(f'  Epoch {ep:02d} | {sp:5s} | loss={val:.4f}')

    test_loss, test_acc = run_epoch(test_loader, training=False)
    print(f'\nTest loss: {test_loss:.4f}  |  Test acc: {test_acc:.4f}')
    mlflow.log_metrics({'test_loss': test_loss, 'test_acc': test_acc})
    mlflow.pytorch.log_model(model, artifact_path='model')
    print('\nModelo registrado no MLflow.')

# ── Curvas de convergência ────────────────────────────────────────────────────
# Use the new plot_convergence_curves function from casting_manufacturing_product module
plot_convergence_curves(history, title="Model Training Convergence")

# ── Avaliação no teste ────────────────────────────────────────────────────────
test_loss, test_acc = run_epoch(test_loader, training=False)
print(f"\nTest loss: {test_loss:.4f}  |  Test acc: {test_acc:.4f}")

# ── Kernels da primeira camada ────────────────────────────────────────────────
# Note: plot_kernels is now imported from casting_manufacturing_product module

plot_kernels(model)

# ── Feature maps ─────────────────────────────────────────────────────────────
# Note: plot_feature_maps is now imported from casting_manufacturing_product module

plot_feature_maps(model, test_loader, CLASS_NAMES, layer_idx=2, n_maps=16)
plot_feature_maps(model, test_loader, CLASS_NAMES, layer_idx=6, n_maps=16)
plot_feature_maps(model, test_loader, CLASS_NAMES, layer_idx=10, n_maps=16)

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
# Note: grad_cam, show_grad_cam, and show_predictions_with_gradcam are now imported from casting_manufacturing_product module

show_grad_cam(model, test_loader, CLASS_NAMES)

show_predictions_with_gradcam(model, test_loader, CLASS_NAMES, n=8)

# Note: plot_confusion_matrix is now imported from casting_manufacturing_product module
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plot_confusion_matrix(model, train_loader, CLASS_NAMES, title="Matriz de Confusão — Treino")
plot_confusion_matrix(model, test_loader, CLASS_NAMES, title="Matriz de Confusão — Teste")

# todos os exemplos
show_predictions_with_gradcam(model, test_loader, CLASS_NAMES, n=32)

# somente erros
show_predictions_with_gradcam(model, test_loader, CLASS_NAMES, n=32, only_errors=True)