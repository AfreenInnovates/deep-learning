# training_utils.py
import os
import random
from io import BytesIO
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import requests

# Multiprocessing safety for dataloaders (helps Jupyter/Colab)
try:
    import torch.multiprocessing as _mp
    _mp.set_start_method("spawn", force=True)
except Exception:
    pass

# Helpers
def get_device(preferred: str = "cuda") -> str:
    """Return a device string ('cuda' if available and preferred, else 'cpu')."""
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def denormalize_batch(tensor: torch.Tensor,
                      mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    Undo normalization for a batch tensor in [B, C, H, W] and clamp to [0,1].
    Returns CPU tensor.
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    tensor = tensor.cpu() * std + mean
    return tensor.clamp(0, 1)

# Training function (returns history)
def train(model: torch.nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer_fn: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int = 5,
          device: str = "cuda",
          verbose: bool = True) -> dict:
    """
    Train loop that returns history dict with per-epoch metrics.
    Args:
      model: torch model
      train_dataloader, test_dataloader: DataLoader
      optimizer_fn: optimizer instance
      loss_fn: loss function (e.g., nn.CrossEntropyLoss())
      epochs: number epochs
      device: 'cuda' or 'cpu'
      verbose: print progress
    Returns:
      history: dict with lists 'train_loss','train_acc','val_loss','val_acc'
    """
    device = get_device(device)
    model.to(device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for X, y in tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer_fn.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer_fn.step()

            running_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total += X.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in tqdm(test_dataloader, desc=f"Epoch {epoch}/{epochs} [Eval]", leave=False):
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                outputs = model(X)
                loss = loss_fn(outputs, y)

                val_loss_sum += loss.item() * X.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += X.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Epoch [{epoch}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if verbose:
        print("Training complete~~")
    return history

# Plot training history
def plot_history(history: dict, figsize=(10,4)):
    """
    Plot loss and accuracy curves from history dict returned by `train`.
    """
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], marker='o', label="train_loss")
    plt.plot(history["val_loss"], marker='o', label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history["train_acc"], marker='o', label="train_acc")
    plt.plot(history["val_acc"], marker='o', label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Grid visualization (4x4 default)
def visualize_predictions(model: torch.nn.Module,
                          dataloader: DataLoader,
                          class_names: List[str],
                          device: str = "cuda",
                          num_images: int = 16,
                          denorm_mean=(0.485,0.456,0.406),
                          denorm_std=(0.229,0.224,0.225)):
    """
    Display a square grid of predictions (default 4x4 = 16).
    """
    assert int(num_images**0.5)**2 == num_images, "num_images must be a perfect square (e.g., 4,9,16)"
    device = get_device(device)
    model.to(device)
    model.eval()

    X_batch, y_batch = next(iter(dataloader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    if num_images > len(X_batch):
        num_images = len(X_batch)

    idxs = random.sample(range(len(X_batch)), num_images)
    X_sample = X_batch[idxs]
    y_true = y_batch[idxs]

    with torch.no_grad():
        outputs = model(X_sample)
    preds = outputs.argmax(dim=1)

    X_vis = denormalize_batch(X_sample, mean=denorm_mean, std=denorm_std)

    grid_size = int(num_images**0.5)
    plt.figure(figsize=(grid_size*3, grid_size*3))
    for i in range(num_images):
        ax = plt.subplot(grid_size, grid_size, i+1)
        img = X_vis[i].permute(1,2,0).numpy()
        plt.imshow(img)
        true_label = class_names[y_true[i].item()]
        pred_label = class_names[preds[i].item()]
        color = "green" if y_true[i] == preds[i] else "red"
        plt.title(f"P: {pred_label}\nT: {true_label}", color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Side-by-side actual vs predicted per-row view
def vis_preds_side_by_side(model: torch.nn.Module,
                           dataloader: DataLoader,
                           class_names: List[str],
                           device: str = "cuda",
                           num_images: int = 5,
                           denorm_mean=(0.485,0.456,0.406),
                           denorm_std=(0.229,0.224,0.225)):
    """
    Show num_images rows; each row has two columns: Actual (left) and Predicted (right).
    """
    device = get_device(device)
    model.to(device)
    model.eval()

    X_batch, y_batch = next(iter(dataloader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    num_images = min(num_images, len(X_batch))
    idxs = random.sample(range(len(X_batch)), num_images)
    X_sample = X_batch[idxs]
    y_true = y_batch[idxs]

    with torch.no_grad():
        outputs = model(X_sample)
    preds = outputs.argmax(dim=1)

    X_vis = denormalize_batch(X_sample, mean=denorm_mean, std=denorm_std)

    plt.figure(figsize=(6, num_images * 3))
    for i in range(num_images):
        ax1 = plt.subplot(num_images, 2, 2*i + 1)
        plt.imshow(X_vis[i].permute(1,2,0).numpy())
        plt.title(f"Actual: {class_names[y_true[i].item()]}", fontsize=9)
        plt.axis("off")

        ax2 = plt.subplot(num_images, 2, 2*i + 2)
        plt.imshow(X_vis[i].permute(1,2,0).numpy())
        color = "green" if y_true[i] == preds[i] else "red"
        plt.title(f"Pred: {class_names[preds[i].item()]}", color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Predict single image from URL or local path
def predict_image(model: torch.nn.Module,
                  image_path_or_url: str,
                  class_names: List[str],
                  device: str = "cuda",
                  transform: Optional[transforms.Compose] = None,
                  top_k: int = 1) -> Tuple[List[Tuple[str, float]], Image.Image]:
    """
    Predict single image. Returns list of (class_name, probability) up to top_k and the PIL image.
    Example:
        preds, pil_img = predict_image(model, "https://...", class_names, device=device, top_k=3)
    """
    device = get_device(device)
    model.to(device)
    model.eval()

    if image_path_or_url.startswith("http"):
        resp = requests.get(image_path_or_url)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(image_path_or_url).convert("RGB")

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)

    topk = min(top_k, probs.shape[0])
    top_probs, top_idx = torch.topk(probs, topk)
    results = [(class_names[idx.item()], float(top_probs[i].item())) for i, idx in enumerate(top_idx)]

    return results, img

