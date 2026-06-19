"""Synthetic datasets used to illustrate principal curves.

The examples are intentionally diverse: compact Gaussian blobs, non-convex
manifolds, spirals, rings, and bridge-like structures. They are useful for
visualizing what a single global principal curve can and cannot represent.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

Dataset = Tuple[np.ndarray, np.ndarray]


def generate_datasets_2d(n_samples: int = 1000, random_state: int = 0) -> Dict[str, Dataset]:
    """Generate several two-dimensional synthetic datasets.

    Parameters
    n_samples:
        Number of points per dataset.
    random_state:
        Seed for reproducible data generation.

    Returns
    dict
        Ordered mapping ``name -> (X, y)`` where ``X`` has shape
        ``(n_samples, 2)`` and ``y`` is a synthetic label vector used only for
        visualization.
    """

    rng = np.random.default_rng(random_state)
    datasets: Dict[str, Dataset] = OrderedDict()

    X, y = make_blobs(
        n_samples=n_samples,
        centers=[[-2.0, 0.0], [2.0, 0.0]],
        cluster_std=0.9,
        random_state=random_state,
    )
    datasets["two_gaussians"] = (X, y)

    X, y = make_blobs(
        n_samples=n_samples,
        centers=3,
        cluster_std=0.59,
        random_state=random_state,
    )
    datasets["three_gaussians"] = (X, y)

    X, y = make_moons(n_samples=n_samples, noise=0.06, random_state=random_state)
    datasets["two_moons"] = (X, y)

    X, y = make_circles(n_samples=n_samples, noise=0.05, factor=0.4, random_state=random_state)
    datasets["concentric_circles"] = (X, y)

    X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.7, random_state=random_state)
    transform = np.array([[0.6, -0.8], [1.8, 0.4]])
    datasets["anisotropic_blobs"] = (X @ transform, y)

    n = n_samples
    t = np.linspace(0.4, 4.5 * np.pi, n)
    r = 0.18 * t
    X = np.column_stack([r * np.cos(t), r * np.sin(t)])
    X += 0.12 * rng.normal(size=X.shape)
    y = np.zeros(n, dtype=int)
    datasets["spiral"] = (X, y)

    n1 = n_samples // 2
    n2 = n_samples - n1
    t1 = np.linspace(0.5, 3.8 * np.pi, n1)
    r1 = 0.18 * t1
    X1 = np.column_stack([r1 * np.cos(t1), r1 * np.sin(t1)])
    t2 = np.linspace(0.5, 3.8 * np.pi, n2)
    r2 = 0.18 * t2
    X2 = np.column_stack([r2 * np.cos(t2 + np.pi), r2 * np.sin(t2 + np.pi)])
    X = np.vstack([X1, X2]) + 0.10 * rng.normal(size=(n_samples, 2))
    y = np.array([0] * n1 + [1] * n2)
    datasets["two_spirals"] = (X, y)

    x = np.linspace(-4, 4, n_samples)
    y_signal = np.sin(1.5 * x)
    X = np.column_stack([x, y_signal]) + 0.15 * rng.normal(size=(n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    datasets["noisy_sine"] = (X, y)

    theta = np.linspace(0.05, np.pi - 0.05, n_samples)
    radius = 2.0
    X = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    X += 0.12 * rng.normal(size=X.shape)
    y = np.zeros(n_samples, dtype=int)
    datasets["u_shape"] = (X, y)

    n_blob = int(0.45 * n_samples)
    n_bridge = n_samples - 2 * n_blob
    X_left = rng.normal(loc=[-2.5, 0.0], scale=[0.55, 0.55], size=(n_blob, 2))
    X_right = rng.normal(loc=[2.5, 0.0], scale=[0.55, 0.55], size=(n_blob, 2))
    xb = np.linspace(-1.8, 1.8, n_bridge)
    yb = 0.12 * rng.normal(size=n_bridge)
    X_bridge = np.column_stack([xb, yb])
    X = np.vstack([X_left, X_right, X_bridge])
    y = np.array([0] * n_blob + [1] * n_blob + [2] * n_bridge)
    datasets["two_blobs_with_bridge"] = (X, y)

    n1 = n_samples // 2
    n2 = n_samples - n1
    X_horizontal = np.column_stack([rng.normal(0, 1.7, size=n1), rng.normal(0, 0.12, size=n1)])
    X_vertical = np.column_stack([rng.normal(0, 0.12, size=n2), rng.normal(0, 1.7, size=n2)])
    X = np.vstack([X_horizontal, X_vertical])
    y = np.array([0] * n1 + [1] * n2)
    datasets["cross"] = (X, y)

    n_ring = int(0.7 * n_samples)
    n_center = n_samples - n_ring
    theta = rng.uniform(0, 2 * np.pi, size=n_ring)
    radius = rng.normal(2.0, 0.12, size=n_ring)
    X_ring = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    X_center = rng.normal(loc=[0.0, 0.0], scale=[0.35, 0.35], size=(n_center, 2))
    X = np.vstack([X_ring, X_center])
    y = np.array([0] * n_ring + [1] * n_center)
    datasets["ring_plus_center"] = (X, y)

    return datasets


def generate_datasets_3d(n_samples: int = 1200, random_state: int = 0) -> Dict[str, Dataset]:
    """Generate several three-dimensional synthetic datasets."""

    rng = np.random.default_rng(random_state)
    datasets: Dict[str, Dataset] = OrderedDict()

    X, y = make_blobs(
        n_samples=n_samples,
        centers=[[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
        cluster_std=0.85,
        random_state=random_state,
    )
    datasets["two_gaussians_3d"] = (X, y)

    X, y = make_blobs(
        n_samples=n_samples,
        centers=[[-2, 0, 0], [2, 0, 0], [0, 2.8, 2.2]],
        cluster_std=0.7,
        random_state=random_state,
    )
    datasets["three_gaussians_3d"] = (X, y)

    X, y = make_blobs(
        n_samples=n_samples,
        centers=[[-2, 0, 0], [2, 0, 0], [0, 2.5, 2.0]],
        cluster_std=0.7,
        random_state=random_state,
    )
    transform = np.array([[1.2, -0.7, 0.2], [0.4, 1.6, -0.5], [0.2, 0.6, 1.3]])
    datasets["anisotropic_blobs_3d"] = (X @ transform, y)

    t = np.linspace(0, 4 * np.pi, n_samples)
    X = np.column_stack([np.cos(t), np.sin(t), t / (2 * np.pi)])
    X += 0.10 * rng.normal(size=X.shape)
    y = np.zeros(n_samples, dtype=int)
    datasets["helix"] = (X, y)

    n1 = n_samples // 2
    n2 = n_samples - n1
    t1 = np.linspace(0, 4 * np.pi, n1)
    X1 = np.column_stack([np.cos(t1), np.sin(t1), t1 / (2 * np.pi)])
    t2 = np.linspace(0, 4 * np.pi, n2)
    X2 = np.column_stack([np.cos(t2 + np.pi), np.sin(t2 + np.pi), t2 / (2 * np.pi)])
    X = np.vstack([X1, X2]) + 0.08 * rng.normal(size=(n_samples, 3))
    y = np.array([0] * n1 + [1] * n2)
    datasets["double_helix"] = (X, y)

    t = np.linspace(-4, 4, n_samples)
    X = np.column_stack([t, np.sin(1.5 * t), 0.5 * np.cos(0.8 * t)])
    X += 0.12 * rng.normal(size=X.shape)
    y = np.zeros(n_samples, dtype=int)
    datasets["sine_3d"] = (X, y)

    theta = np.linspace(0.05, np.pi - 0.05, n_samples)
    X = np.column_stack([2.0 * np.cos(theta), 2.0 * np.sin(theta), 0.8 * np.sin(2 * theta)])
    X += 0.10 * rng.normal(size=X.shape)
    y = np.zeros(n_samples, dtype=int)
    datasets["u_shape_3d"] = (X, y)

    t = np.linspace(0, 2 * np.pi, n_samples)
    circle = np.column_stack([2 * np.cos(t), 2 * np.sin(t), np.zeros_like(t)])
    rotation = np.array([[1, 0, 0], [0, np.cos(np.pi / 5), -np.sin(np.pi / 5)], [0, np.sin(np.pi / 5), np.cos(np.pi / 5)]])
    X = circle @ rotation.T + 0.08 * rng.normal(size=(n_samples, 3))
    y = np.zeros(n_samples, dtype=int)
    datasets["tilted_circle_3d"] = (X, y)

    n_blob = int(0.42 * n_samples)
    n_bridge = n_samples - 2 * n_blob
    X_left = rng.normal(loc=[-2.8, 0.0, 0.0], scale=[0.55, 0.55, 0.55], size=(n_blob, 3))
    X_right = rng.normal(loc=[2.8, 0.0, 0.0], scale=[0.55, 0.55, 0.55], size=(n_blob, 3))
    tb = np.linspace(-2.0, 2.0, n_bridge)
    X_bridge = np.column_stack([tb, 0.2 * np.sin(2 * tb), 0.15 * rng.normal(size=n_bridge)])
    X = np.vstack([X_left, X_right, X_bridge])
    y = np.array([0] * n_blob + [1] * n_blob + [2] * n_bridge)
    datasets["two_blobs_with_bridge_3d"] = (X, y)

    return datasets
