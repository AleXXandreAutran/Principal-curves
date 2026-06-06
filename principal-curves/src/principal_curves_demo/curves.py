"""A simple iterative approximation of principal curves.

This implementation is deliberately lightweight and educational. It is not the
full density-ridge method of Ozertem/Erdogmus or the manifold traversal method
based on geodesics. It approximates a single global central curve using local
averaging and smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


@dataclass
class PrincipalCurveResult:
    """Result returned by :func:`principal_curve`."""

    curve: np.ndarray
    history: List[np.ndarray]


def initialize_curve_pca(X: np.ndarray, n_nodes: int = 40) -> np.ndarray:
    """Initialize curve nodes along the first principal component.

    Parameters
    ----------
    X:
        Data matrix with shape ``(n_samples, n_features)``.
    n_nodes:
        Number of nodes used to discretize the curve.

    Returns
    -------
    np.ndarray
        Initial curve with shape ``(n_nodes, n_features)``.
    """

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2.")

    pca = PCA(n_components=1)
    t = pca.fit_transform(X)[:, 0]
    t_nodes = np.linspace(t.min(), t.max(), n_nodes)
    direction = pca.components_[0]
    center = X.mean(axis=0)
    return center + np.outer(t_nodes, direction)


def assign_points_to_curve_nodes(X: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Assign each data point to the nearest curve node."""

    distances = cdist(X, curve)
    return np.argmin(distances, axis=1)


def update_curve_nodes(X: np.ndarray, curve: np.ndarray, labels: np.ndarray, bandwidth: float = 3.0) -> np.ndarray:
    """Move each node toward a local weighted average of data points.

    The weight of a point decreases with the distance between the point's
    assigned node index and the node being updated.
    """

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    n_nodes = len(curve)
    new_curve = np.zeros_like(curve)

    for k in range(n_nodes):
        index_distance = labels - k
        weights = np.exp(-0.5 * (index_distance / bandwidth) ** 2)

        if weights.sum() > 1e-12:
            new_curve[k] = np.sum(X * weights[:, None], axis=0) / weights.sum()
        else:
            new_curve[k] = curve[k]

    return new_curve


def smooth_curve(curve: np.ndarray, smoothness: float = 0.35) -> np.ndarray:
    """Smooth a polyline by moving internal nodes toward neighbor averages."""

    if not 0 <= smoothness <= 1:
        raise ValueError("smoothness must lie in [0, 1].")

    new_curve = curve.copy()
    for k in range(1, len(curve) - 1):
        local_mean = 0.5 * (curve[k - 1] + curve[k + 1])
        new_curve[k] = (1.0 - smoothness) * curve[k] + smoothness * local_mean
    return new_curve


def principal_curve(
    X: np.ndarray,
    n_nodes: int = 45,
    n_iter: int = 40,
    bandwidth: float = 3.0,
    smoothness: float = 0.35,
) -> PrincipalCurveResult:
    """Approximate one global principal curve.

    Algorithm
    ---------
    1. Initialize nodes with PCA.
    2. Assign every data point to its closest node.
    3. Move each node to a local weighted average of the data.
    4. Smooth the curve.
    5. Repeat steps 2--4.

    Notes
    -----
    This method learns a single global curve. On separated clusters, it may
    connect components artificially. This behavior is useful pedagogically for
    contrasting simple principal curves with density-ridge and clustering
    methods.
    """

    if n_iter < 0:
        raise ValueError("n_iter must be non-negative.")

    curve = initialize_curve_pca(X, n_nodes=n_nodes)
    history: List[np.ndarray] = [curve.copy()]

    for _ in range(n_iter):
        labels = assign_points_to_curve_nodes(X, curve)
        curve = update_curve_nodes(X, curve, labels, bandwidth=bandwidth)
        curve = smooth_curve(curve, smoothness=smoothness)
        history.append(curve.copy())

    return PrincipalCurveResult(curve=curve, history=history)
