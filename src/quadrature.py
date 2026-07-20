"""Gauss-Legendre quadrature helpers for two-dimensional integration.

Provides nodes and weights on [-1, 1] together with a mapping to an
arbitrary interval [a, b], and a convenience function that builds
a full two-dimensional quadrature grid from two marginal distributions.
"""

from __future__ import annotations

import numpy as np

from .marginals import MarginalDistribution


def gauss_legendre_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights on ``[-1, 1]``.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    xi : ndarray, shape (n,)
        Nodes on ``[-1, 1]``.
    w : ndarray, shape (n,)
        Corresponding quadrature weights.
    """
    xi, w = np.polynomial.legendre.leggauss(n)
    return xi, w


def map_to_interval(xi: np.ndarray, w: np.ndarray,
                    a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Affinely map Gauss-Legendre nodes and weights from ``[-1, 1]`` to ``[a, b]``.

    The change-of-variable is  x = (b-a)/2 * xi + (b+a)/2,
    and the weight scaling factor is  (b-a)/2.

    Parameters
    ----------
    xi : ndarray
        GL nodes on ``[-1, 1]``.
    w : ndarray
        GL weights on ``[-1, 1]``.
    a : float
        Left endpoint of target interval.
    b : float
        Right endpoint of target interval.

    Returns
    -------
    x_mapped : ndarray
        Nodes on ``[a, b]``.
    w_mapped : ndarray
        Scaled weights (sum to ``b - a``).
    """
    half_len  = (b - a) / 2.0
    mid       = (b + a) / 2.0
    x_mapped  = half_len * xi + mid
    w_mapped  = half_len * w
    return x_mapped, w_mapped


def build_quadrature_grid(marginal_X: MarginalDistribution,
                          marginal_Y: MarginalDistribution,
                          n: int = 200,
                          q: float = 1e-4) -> dict:
    """Build a two-dimensional Gauss-Legendre quadrature grid.

    Each axis uses *n* nodes mapped to the effective support of the
    corresponding marginal distribution (defined by the *q* and *1-q*
    quantiles).

    Parameters
    ----------
    marginal_X : MarginalDistribution
        Marginal for the first asset (rows of the joint grid).
    marginal_Y : MarginalDistribution
        Marginal for the second asset (columns of the joint grid).
    n : int
        Number of GL nodes per dimension (default 200).
    q : float
        Tail-truncation probability used to define the support
        (default 1e-4).

    Returns
    -------
    dict with keys
        ``x_nodes``   : ndarray (n,) — quadrature nodes for X
        ``y_nodes``   : ndarray (n,) — quadrature nodes for Y
        ``x_weights`` : ndarray (n,) — GL weights for X
        ``y_weights`` : ndarray (n,) — GL weights for Y
        ``x_low``, ``x_high`` : float — support bounds for X
        ``y_low``, ``y_high`` : float — support bounds for Y
    """
    x_low, x_high = marginal_X.support()
    y_low, y_high = marginal_Y.support()

    xi, w = gauss_legendre_nodes_weights(n)

    x_nodes, x_weights = map_to_interval(xi, w, x_low, x_high)
    y_nodes, y_weights = map_to_interval(xi, w, y_low, y_high)

    return {
        "x_nodes":   x_nodes,
        "y_nodes":   y_nodes,
        "x_weights": x_weights,
        "y_weights": y_weights,
        "x_low":  x_low,
        "x_high": x_high,
        "y_low":  y_low,
        "y_high": y_high,
    }
