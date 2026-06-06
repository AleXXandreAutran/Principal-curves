"""Plotting utilities for principal curves."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .curves import principal_curve

Dataset = Tuple[np.ndarray, np.ndarray]


def plot_principal_curves_2d(
    datasets: Dict[str, Dataset],
    n_nodes: int = 55,
    n_iter: int = 60,
    bandwidth: float = 3.0,
    smoothness: float = 0.35,
    n_cols: int = 3,
    output_path: str | Path | None = None,
):
    """Plot one principal curve for each 2D dataset."""

    n_rows = int(np.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.8 * n_rows))
    axes = np.ravel(axes)

    for ax, (name, (X, y)) in zip(axes, datasets.items()):
        result = principal_curve(X, n_nodes=n_nodes, n_iter=n_iter, bandwidth=bandwidth, smoothness=smoothness)
        curve = result.curve

        ax.scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="tab10", alpha=0.65)
        ax.plot(curve[:, 0], curve[:, 1], "k-", linewidth=2.3, label="principal curve")
        ax.scatter(curve[:, 0], curve[:, 1], c="red", s=28, marker="x", label="nodes")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name.replace("_", " "))
        ax.legend(loc="best", fontsize=8)

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.suptitle(
        "Approximation of principal curves on synthetic geometries\n"
        f"n_nodes={n_nodes}, n_iter={n_iter}, bandwidth={bandwidth}, smoothness={smoothness}",
        fontsize=14,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")

    return fig


def plot_principal_curves_3d(
    datasets: Dict[str, Dataset],
    n_nodes: int = 55,
    n_iter: int = 60,
    bandwidth: float = 3.0,
    smoothness: float = 0.35,
    n_cols: int = 3,
    output_path: str | Path | None = None,
):
    """Plot one principal curve for each 3D dataset."""

    n_rows = int(np.ceil(len(datasets) / n_cols))
    fig = plt.figure(figsize=(6.2 * n_cols, 5.2 * n_rows))

    for idx, (name, (X, y)) in enumerate(datasets.items(), start=1):
        ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
        result = principal_curve(X, n_nodes=n_nodes, n_iter=n_iter, bandwidth=bandwidth, smoothness=smoothness)
        curve = result.curve

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=6, cmap="tab10", alpha=0.55)
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "k-", linewidth=2.2, label="principal curve")
        ax.scatter(curve[:, 0], curve[:, 1], curve[:, 2], c="red", s=20, marker="x", label="nodes")
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "Approximation of principal curves on 3D synthetic geometries\n"
        f"n_nodes={n_nodes}, n_iter={n_iter}, bandwidth={bandwidth}, smoothness={smoothness}",
        fontsize=14,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")

    return fig
