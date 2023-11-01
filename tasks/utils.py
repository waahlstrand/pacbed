import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from typing import *
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch
import pandas as pd


def visualize_classification_metrics_csv(csv_path: Path, **kwargs) -> Tuple[matplotlib.figure.Figure, plt.Axes, pd.DataFrame]:

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(1, 1, **kwargs)
    
    train_loss = df[["step", "train_loss_epoch"]].dropna()
    val_loss = df[["step", "val_loss"]].dropna()

    ax.semilogy(train_loss["step"], train_loss["train_loss_epoch"], label="training loss")
    ax.semilogy(val_loss["step"], val_loss["val_loss"], label="validation loss")

    if 'train_acc' in df.columns:
        ax.plot(df['step'], df['train_acc'], label='train_acc')
    
    if 'val_acc' in df.columns:
        ax.plot(df['step'], df['val_acc'], label='val_acc')

    ax.legend()
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')

    return fig, ax, train_loss, val_loss


def scatter_hist(x: np.ndarray, y: np.ndarray, left: float = 0.1, width: float = 0.65, bottom: float = 0.1, height: float = 0.65, spacing: float = 0.005) -> Tuple[matplotlib.figure.Figure, plt.Axes, plt.Axes, plt.Axes]:

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha=0.8)

    # Compute domain
    xmin = np.min(x)
    xmax = np.max(x)

    # Perfect line
    ax.plot([xmin, xmin], [xmax,xmax], 'k')

    # now determine nice limits by hand:
    # binwidth = 0.25
    # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    # lim = (int(xymax/binwidth) + 1) * binwidth
    

    # bins = np.arange(-lim, lim + binwidth, binwidth)
    
    ax_histx.hist(x.flatten())
    ax_histy.hist(y.flatten(), orientation='horizontal')
    
    return fig, ax, ax_histx, ax_histy


def preprocess(x: np.ndarray, scaling=0.25, size=256):

    # Reshape 
    x = x.reshape((size, size))

    x = np.abs(scaling * x / np.mean(np.abs(x)))

    return x


def bin_to_image(file: Path):

    # Read binary file
    with open(file, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)

    # Preprocess
    data = preprocess(data)

    return data


def experimental_dataloader(data_dir: Path, results_file: Path, **kwargs):

    with open(results_file, "r") as f:
        results = json.load(f)

    x  = torch.stack([torch.tensor(bin_to_image(data_dir / result['file'])) for result in results['data']]).unsqueeze(1)
    y  = torch.tensor([result['thickness'] for result in results['data']]).log()

    dataset = TensorDataset(x, y)

    return DataLoader(dataset, **kwargs)


def plot_image_with_predictions(
        x: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        n_samples: int = 4,
):
    """
    Plot a sample from a batch with its ground truth and predictions

    Args:
        x: Input image, shape (B, 1, H, W)
        y: Ground truth, shape (B, 1)
        y_hat: Predictions, shape (B, 1)
        save_path: Path to save the plot
    """
    assert n_samples % 2 == 0, "n_samples must be even"

    fig, ax = plt.subplots(nrows=1, ncols=n_samples, figsize=(15, 5))

    for i in range(n_samples):

        # Plot each image with both its ground truth and predictions 
        # in the title
        ax[i].imshow(x[i, 0], cmap="gray")
        ax[i].set_title(f"y: {y[i].item():.2f}\ny_hat: {y_hat[i].item():.2f}")
        ax[i].axis("off")
        ax[i].set_aspect("equal")
        ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

    return fig, ax
    


        


        



