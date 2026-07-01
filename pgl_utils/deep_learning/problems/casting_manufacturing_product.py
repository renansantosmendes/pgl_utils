"""
Módulo para classificação de produtos de manufatura por fundição.

Este módulo contém classes de transformação de imagens e funções de visualização
para análise de modelos de deep learning aplicados à detecção de defeitos em
produtos manufaturados.

Variáveis globais esperadas:
    DEVICE: torch.device - Dispositivo para computação (CPU ou CUDA)
    IMG_SIZE: int - Tamanho das imagens (altura e largura)
    CLASS_NAMES: list - Lista com os nomes das classes
    model: torch.nn.Module - Modelo PyTorch treinado
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================================
# CLASSES DE TRANSFORMAÇÃO E AUGMENTAÇÃO
# ============================================================================

from typing import List
from PIL.Image import Image
from torch.utils.data import DataLoader

class CropCircular:
    """
    Crop the circular region of the casting part from an image, removing the background.
    
    Uses Otsu thresholding and morphological operations to identify the part
    and create a centered square crop with additional margin.
    
    Args:
        margin: Proportional margin around the part (default: 0.05 = 5%)
    """

    def __init__(self, margin: float = 0.05) -> None:
        self.margin = margin

    def __call__(self, img: Image) -> Image:
        """
        Apply the circular crop to the image.
        
        Args:
            img: PIL Image - Input image
            
        Returns:
            PIL Image - Cropped image
        """
        from skimage.filters import threshold_otsu
        from skimage.morphology import binary_closing, disk

        arr = np.array(img.convert('L'))
        h, w = arr.shape

        # Otsu threshold + closing to fill holes
        thresh = threshold_otsu(arr)
        binary = binary_closing(arr > thresh, disk(5)).astype(np.uint8)

        # Bounding box of the part's pixels
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
        else:
            rmin, rmax, cmin, cmax = 0, h, 0, w

        # Additional margin
        pad = int(min(h, w) * self.margin)
        rmin = max(0, rmin - pad)
        rmax = min(h, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(w, cmax + pad)

        # Centered square crop
        side = max(rmax - rmin, cmax - cmin)
        cy = (rmin + rmax) // 2
        cx = (cmin + cmax) // 2
        rmin = max(0, cy - side // 2)
        rmax = min(h, rmin + side)
        cmin = max(0, cx - side // 2)
        cmax = min(w, cmin + side)

        return img.crop((cmin, rmin, cmax, rmax))


class CutOut:
    """
    CutOut data augmentation: randomly masks out rectangular regions of the image.
    
    Args:
        n_holes: Number of regions to remove
        length: Size (side length) of each square region
    """

    def __init__(self, n_holes: int = 1, length: int = 32) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply CutOut on the tensor image.
        
        Args:
            img: torch.Tensor - Image tensor (C, H, W)
            
        Returns:
            torch.Tensor - Image with masked regions
        """
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones_like(img)

        for _ in range(self.n_holes):
            cy = torch.randint(h, (1,)).item()
            cx = torch.randint(w, (1,)).item()
            y1 = max(0, cy - self.length // 2)
            y2 = min(h, cy + self.length // 2)
            x1 = max(0, cx - self.length // 2)
            x2 = min(w, cx + self.length // 2)
            mask[:, y1:y2, x1:x2] = 0

        return img * mask


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO DE DADOS
# ============================================================================

def show_examples(loader: DataLoader, class_names: List[str], n: int = 4) -> None:
    """
    Display example images per class.
    
    Args:
        loader: DataLoader with the images
        class_names: List of class names
        n: Number of examples per class
    """
    images, labels = next(iter(loader))
    images = images * 0.5 + 0.5

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for cls_idx, cls_name in enumerate(class_names):
        idxs = (labels == cls_idx).nonzero(as_tuple=True)[0][:n]
        for col, i in enumerate(idxs):
            ax = axes[cls_idx][col]
            ax.imshow(images[i].squeeze(), cmap="gray")
            ax.set_title(cls_name)
            ax.axis("off")
    
    plt.suptitle("Exemplos: defeituoso (topo) vs. ok (baixo)")
    plt.tight_layout()
    plt.show()


def show_normalized(loader: DataLoader, class_names: List[str], n: int = 6) -> None:
    """
    Display normalized images and their histograms per class.
    
    Args:
        loader: DataLoader with normalized images
        class_names: List of class names
        n: Number of examples per class
    """
    images, labels = next(iter(loader))
    n_cols = n

    fig, axes = plt.subplots(
        len(class_names) * 2, n_cols,
        figsize=(n_cols * 2.2, len(class_names) * 2 * 2.2)
    )

    for cls_idx, cls_name in enumerate(class_names):
        idxs = (labels == cls_idx).nonzero(as_tuple=True)[0][:n]
        for col, i in enumerate(idxs):
            img = images[i].squeeze()
            row_raw = cls_idx * 2
            row_norm = row_raw + 1

            # Normalized image (values can be negative)
            vmin, vmax = img.min().item(), img.max().item()
            axes[row_raw][col].imshow(
                img.numpy(), cmap='gray', vmin=vmin, vmax=vmax
            )
            axes[row_raw][col].set_title(
                cls_name.replace('_front', '') + ' (norm)',
                fontsize=7
            )
            axes[row_raw][col].axis('off')

            # Intensity histogram
            axes[row_norm][col].hist(
                img.numpy().flatten(), bins=40,
                color='steelblue', edgecolor='none'
            )
            axes[row_norm][col].axvline(0, color='red', linewidth=1)
            axes[row_norm][col].set_title(
                f'm={img.mean():.2f} s={img.std():.2f}',
                fontsize=6
            )
            axes[row_norm][col].tick_params(labelsize=5)

    plt.suptitle(
        'Normalized images | odd=image  even=histogram  red=zero',
        fontsize=8
    )
    plt.tight_layout(pad=0.3)
    plt.show()


# ============================================================================
# FUNÇÕES AUXILIARES PARA GRAD-CAM
# ============================================================================

import torch
from torch.nn import Module
from torch import Tensor
from typing import Tuple
import numpy as np

def _get_grad_cam(model: Module, img_tensor: Tensor, device: torch.device, img_size: int) -> np.ndarray:
    """
    Calculate Grad-CAM for a given image (internal helper function).
    
    Args:
        model: PyTorch model
        img_tensor: Image tensor
        device: Computation device (CPU/CUDA)
        img_size: Image size for resizing
        
    Returns:
        numpy.ndarray: Grad-CAM activation map
    """
    grads_store, acts_store = {}, {}

    def fwd(m: Module, inp: Tuple, out: Tensor) -> None:
        acts_store["v"] = out

    def bwd(m: Module, gi: Tuple, go: Tuple) -> None:
        grads_store["v"] = go[0]

    last_conv = model.features[-3]
    h1 = last_conv.register_forward_hook(fwd)
    h2 = last_conv.register_full_backward_hook(bwd)

    model.zero_grad()
    out = model(img_tensor.unsqueeze(0).to(device))
    out[0, out.argmax(1).item()].backward()

    h1.remove()
    h2.remove()

    g = grads_store["v"][0]
    a = acts_store["v"][0]
    cam = torch.nn.functional.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False
    )
    return cam.squeeze().detach().cpu().numpy()


def grad_cam(model: Module, img_tensor: Tensor, target_class: int) -> np.ndarray:
    """
    Calculate Grad-CAM for a specific class.
    
    Args:
        model: PyTorch model
        img_tensor: Image tensor (1, C, H, W)
        target_class: Target class index
        
    Returns:
        numpy.ndarray: Grad-CAM activation map
        
    Note:
        Requires global DEVICE variable to be set.
    """
    model.eval()
    gradients, activations = {}, {}

    def fwd_hook(m: Module, inp: Tuple, out: Tensor) -> None:
        activations["feat"] = out

    def bwd_hook(m: Module, grad_in: Tuple, grad_out: Tuple) -> None:
        gradients["feat"] = grad_out[0]

    last_conv = model.features[-3]
    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_full_backward_hook(bwd_hook)

    output = model(img_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    h1.remove()
    h2.remove()

    grads = gradients["feat"][0]
    acts = activations["feat"][0]
    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(0)
    cam = torch.nn.functional.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    return cam.detach().cpu().numpy()


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO COM GRAD-CAM
# ============================================================================

from typing import Optional
def show_grad_cam(model: torch.nn.Module, loader: DataLoader, class_names: List[str], n: int = 4) -> None:
    """
    Display original images and their Grad-CAM maps.
    
    Args:
        model: Trained PyTorch model
        loader: DataLoader with images
        class_names: List of class names
        n: Number of examples to display
    
    Note:
        Requires global DEVICE and IMG_SIZE variables.
    """
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for col in range(n):
        img = images[col:col + 1].to(DEVICE)
        label = labels[col].item()
        cam = grad_cam(model, img, label)

        img_np = (images[col].squeeze().numpy() * 0.5 + 0.5)

        cam_resized = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = torch.nn.functional.interpolate(
            cam_resized, size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False
        )
        cam_resized = cam_resized.squeeze().numpy()

        axes[0][col].imshow(img_np, cmap="gray")
        axes[0][col].set_title(class_names[label])
        axes[0][col].axis("off")

        axes[1][col].imshow(img_np, cmap="gray")
        axes[1][col].imshow(cam_resized, cmap="jet", alpha=0.45)
        axes[1][col].set_title("Grad-CAM")
        axes[1][col].axis("off")

    plt.suptitle("Original image (top) vs Grad-CAM (bottom)")
    plt.tight_layout()
    plt.show()


def show_predictions_with_gradcam(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: List[str],
    n: int = 32,
    only_errors: bool = False
) -> None:
    """
    Display model predictions with Grad-CAM visualization.
    
    Args:
        model: Trained PyTorch model
        loader: DataLoader with images
        class_names: List of class names
        n: Maximum number of examples to display
        only_errors: If True, display only incorrect predictions
    
    Note:
        Requires global DEVICE and IMG_SIZE variables.
    """
    model.eval()
    all_images, all_labels, all_preds, all_cams = [], [], [], []

    # Collect images until n examples (or only errors) are gathered
    for images, labels in loader:
        with torch.no_grad():
            preds = model(images.to(DEVICE)).argmax(dim=1).cpu()

        for i in range(len(images)):
            correct = labels[i].item() == preds[i].item()
            if only_errors and correct:
                continue
            
            all_images.append(images[i])
            all_labels.append(labels[i].item())
            all_preds.append(preds[i].item())
            all_cams.append(_get_grad_cam(model, images[i], DEVICE, IMG_SIZE))
            
            if len(all_images) >= n:
                break
        if len(all_images) >= n:
            break

    n = len(all_images)
    n_cols = 8
    n_rows = (n + n_cols - 1) // n_cols

    # Each example occupies 2 rows: image + grad-cam
    fig, axes = plt.subplots(
        n_rows * 2, n_cols,
        figsize=(n_cols * 1.8, n_rows * 2 * 1.8)
    )
    axes = axes.reshape(n_rows * 2, n_cols)

    for i in range(n):
        row_img = (i // n_cols) * 2
        row_cam = row_img + 1
        col = i % n_cols

        img_np = (all_images[i].squeeze().numpy() * 0.5 + 0.5)
        cam_np = all_cams[i]
        correct = all_labels[i] == all_preds[i]
        color = "green" if correct else "red"
        real_name = class_names[all_labels[i]].replace("_front", "")
        pred_name = class_names[all_preds[i]].replace("_front", "")

        axes[row_img][col].imshow(img_np, cmap="gray")
        axes[row_img][col].set_title(
            f"R:{real_name}\nP:{pred_name}",
            fontsize=5.5, color=color, pad=1
        )
        axes[row_img][col].axis("off")

        axes[row_cam][col].imshow(img_np, cmap="gray")
        axes[row_cam][col].imshow(cam_np, cmap="jet", alpha=0.5)
        axes[row_cam][col].axis("off")

    for i in range(n, n_rows * n_cols):
        axes[(i // n_cols) * 2][i % n_cols].axis("off")
        axes[(i // n_cols) * 2 + 1][i % n_cols].axis("off")

    title = "Only errors" if only_errors else "Prediction vs. Real"
    plt.suptitle(
        title + "  (green=correct | red=error)  R=real  P=pred",
        fontsize=8, y=1.01
    )
    plt.tight_layout(pad=0.3)
    plt.show()


from typing import Optional
def inspect_batch(
    images: torch.Tensor, labels: torch.Tensor, losses: torch.Tensor, outputs: torch.Tensor,
    epoch: int, split: str, batch_idx: int,
    model: torch.nn.Module, device: torch.device, img_size: int, class_names: List[str],
    inspect_n: int = 16
) -> None:
    """
    Inspect the worst predictions of a batch with Grad-CAM.
    
    Args:
        images: Tensor with batch images
        labels: Tensor with true labels
        losses: Tensor with individual losses
        outputs: Tensor with model outputs
        epoch: Current epoch number
        split: Split name ('train'/'val'/'test')
        batch_idx: Batch index
        model: PyTorch model
        device: Computation device (CPU/CUDA)
        img_size: Image size
        class_names: List with class names
        inspect_n: Number of worst examples to display
    """
    top_k = min(inspect_n, len(losses))
    top_idx = losses.argsort(descending=True)[:top_k]

    imgs_dev = images[top_idx]
    imgs_show = imgs_dev.cpu() * 0.5 + 0.5
    lbls = labels[top_idx].cpu()
    lss = losses[top_idx].cpu()
    preds = outputs[top_idx].argmax(dim=1).cpu()

    n_cols = 8
    n_rows = (top_k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows * 2, n_cols,
        figsize=(n_cols * 1.8, n_rows * 2 * 1.8)
    )
    axes = axes.reshape(n_rows * 2, n_cols)

    model.eval()
    for i in range(top_k):
        row_img = (i // n_cols) * 2
        row_cam = row_img + 1
        col = i % n_cols

        img_np = imgs_show[i].squeeze().numpy()
        cam_np = _get_grad_cam(model, imgs_dev[i].cpu(), device, img_size)
        correct = lbls[i].item() == preds[i].item()
        color = "green" if correct else "red"
        real_name = class_names[lbls[i].item()].replace("_front", "")
        pred_name = class_names[preds[i].item()].replace("_front", "")

        axes[row_img][col].imshow(img_np, cmap="gray")
        axes[row_img][col].set_title(
            f"R:{real_name}\nP:{pred_name}\n{lss[i].item():.2f}",
            fontsize=5.5, color=color, pad=1
        )
        axes[row_img][col].axis("off")

        axes[row_cam][col].imshow(img_np, cmap="gray")
        axes[row_cam][col].imshow(cam_np, cmap="jet", alpha=0.5)
        axes[row_cam][col].axis("off")

    for i in range(top_k, n_rows * n_cols):
        axes[(i // n_cols) * 2][i % n_cols].axis("off")
        axes[(i // n_cols) * 2 + 1][i % n_cols].axis("off")

    fig.suptitle(
        f"Spike Ep.{epoch} | {split} | batch {batch_idx} | "
        f"loss={lss.mean().item():.3f}  (verde=ok  vermelho=erro)",
        color="red", fontsize=8
    )
    plt.tight_layout(pad=0.3)
    plt.show()


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO DO MODELO
# ============================================================================

def plot_kernels(model, n_cols=8):
    """
    Visualiza os kernels (filtros) aprendidos pela primeira camada convolucional.
    
    Args:
        model: Modelo PyTorch com camada 'features'
        n_cols: Número de colunas na grade de visualização
    """
    kernels = model.features[0].weight.data.cpu()
    n_filters = kernels.shape[0]
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.2, n_rows * 1.2)
    )
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < n_filters:
            k = kernels[i, 0]
            k = (k - k.min()) / (k.max() - k.min() + 1e-8)
            ax.imshow(k, cmap="viridis")
        ax.axis("off")
    
    plt.suptitle("Kernels aprendidos — Conv2d camada 1")
    plt.tight_layout()
    plt.show()


def plot_feature_maps(model, loader, class_names, layer_idx=2, n_maps=16):
    """
    Visualiza os feature maps de uma camada específica do modelo.
    
    Args:
        model: Modelo PyTorch com camada 'features'
        loader: DataLoader com imagens
        class_names: Lista com nomes das classes
        layer_idx: Índice da camada a visualizar
        n_maps: Número máximo de feature maps a exibir
        
    Note:
        Requer variável global DEVICE definida.
    """
    images, labels = next(iter(loader))
    img = images[0:1].to(DEVICE)

    activation = {}
    def hook_fn(module, inp, out):
        activation["feat"] = out.detach()

    hook = model.features[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(img)
    hook.remove()

    feat = activation["feat"][0].cpu()
    n_maps = min(n_maps, feat.shape[0])
    n_cols = 8
    n_rows = (n_maps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.5, n_rows * 1.5)
    )
    axes = axes.flatten()
    
    for i in range(n_maps):
        fm = feat[i]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        axes[i].imshow(fm, cmap="viridis")
        axes[i].axis("off")
    
    for j in range(n_maps, len(axes)):
        axes[j].axis("off")
    
    plt.suptitle(
        f"Feature maps — camada {layer_idx} | "
        f"classe: {class_names[labels[0].item()]}"
    )
    plt.tight_layout()
    plt.show()


def plot_convergence_curves(history, title="Training Convergence", save_path=None):
    """
    Plots training and validation convergence curves (loss and accuracy).
    
    Args:
        history (dict): Dictionary with training history containing keys:
                       'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title (str): Title for the plot
        save_path (Optional[str]): Path to save the figure. If None, displays instead.
    
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    
    # Check if history has the required keys
    required_keys = {'train_loss', 'val_loss', 'train_acc', 'val_acc'}
    available_keys = set(history.keys())
    
    if not required_keys.issubset(available_keys):
        missing = required_keys - available_keys
        raise ValueError(f"History dictionary missing required keys: {missing}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Get number of epochs
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('Loss Convergence', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)
    
    # Add min loss annotations
    min_train_loss = min(history['train_loss'])
    min_val_loss = min(history['val_loss'])
    min_train_epoch = history['train_loss'].index(min_train_loss) + 1
    min_val_epoch = history['val_loss'].index(min_val_loss) + 1
    
    ax1.annotate(f'Min: {min_train_loss:.4f}', 
                xy=(min_train_epoch, min_train_loss),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, color='blue', alpha=0.7)
    ax1.annotate(f'Min: {min_val_loss:.4f}', 
                xy=(min_val_epoch, min_val_loss),
                xytext=(5, -10), textcoords='offset points',
                fontsize=8, color='red', alpha=0.7)
    
    # Plot accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.set_title('Accuracy Convergence', fontsize=11)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    ax2.set_ylim([0, 105])
    
    # Add max accuracy annotations
    max_train_acc = max(history['train_acc'])
    max_val_acc = max(history['val_acc'])
    max_train_epoch = history['train_acc'].index(max_train_acc) + 1
    max_val_epoch = history['val_acc'].index(max_val_acc) + 1
    
    ax2.annotate(f'Max: {max_train_acc:.1f}%', 
                xy=(max_train_epoch, max_train_acc),
                xytext=(5, -10), textcoords='offset points',
                fontsize=8, color='blue', alpha=0.7)
    ax2.annotate(f'Max: {max_val_acc:.1f}%', 
                xy=(max_val_epoch, max_val_acc),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, color='red', alpha=0.7)
    
    # Add title to figure
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Convergence curves saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(model, loader, class_names, title=""):
    """
    Plota a matriz de confusão para o modelo em um dataset.
    
    Args:
        model: Modelo PyTorch treinado
        loader: DataLoader com imagens
        class_names: Lista com nomes das classes
        title: Título adicional para o gráfico
        
    Note:
        Requer variável global DEVICE definida.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
