"""Simple principal-curve approximation tools for synthetic datasets."""

from .curves import PrincipalCurveResult, principal_curve
from .datasets import generate_datasets_2d, generate_datasets_3d
from .plotting import plot_principal_curves_2d, plot_principal_curves_3d

__all__ = [
    "PrincipalCurveResult",
    "principal_curve",
    "generate_datasets_2d",
    "generate_datasets_3d",
    "plot_principal_curves_2d",
    "plot_principal_curves_3d",
]
