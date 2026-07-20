"""Calibration diagnostics and visualisation.

Provides:
- Bachelier pricing helpers (re-exported from market_data)
- Smile comparison plots (market vs model Bachelier implied vols)
- Joint distribution heatmap
- Calibration error diagnostics (price and vol space)
- Implied Gaussian correlation computation via copula inversion
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe on headless servers
import matplotlib.pyplot as plt

from .market_data import bachelier_call_price, bachelier_implied_vol
from .marginals import MarginalDistribution
from .copulas import build_prior_weights

# Re-export so callers can do `from src.reporting import bachelier_call_price`
__all__ = [
    "bachelier_call_price",
    "bachelier_implied_vol",
    "prices_to_bachelier_vols",
    "compute_calibration_diagnostics",
    "plot_smile_comparison",
    "plot_joint_distribution",
    "gaussian_copula_price",
    "compute_implied_gaussian_correlation",
]

_BPS = 1e-4   # one basis point


# ---------------------------------------------------------------------------
# Vol-space helpers
# ---------------------------------------------------------------------------

def prices_to_bachelier_vols(prices:   np.ndarray,
                              forward:  float,
                              strikes:  np.ndarray,
                              T:        float,
                              tol:      float = 1e-10) -> np.ndarray:
    """Vectorised Bachelier implied-vol inversion.

    Parameters
    ----------
    prices : ndarray
        Observed call prices.
    forward : float
        Forward rate.
    strikes : ndarray
        Corresponding strike grid (same length as *prices*).
    T : float
        Option expiry in years.
    tol : float
        Root-finding tolerance passed to :func:`bachelier_implied_vol`.

    Returns
    -------
    vols : ndarray
        Annualised Bachelier implied vols (NaN on failure).
    """
    prices  = np.asarray(prices,  dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    vols    = np.empty_like(prices)
    for i, (p, k) in enumerate(zip(prices, strikes)):
        try:
            vols[i] = bachelier_implied_vol(p, forward, k, T, tol=tol)
        except Exception:
            vols[i] = np.nan
    return vols


def _model_call_prices(W_ij:     np.ndarray,
                       x_nodes:  np.ndarray,
                       y_nodes:  np.ndarray,
                       strikes:  np.ndarray,
                       asset:    str) -> np.ndarray:
    """Compute model call prices from W_ij without a DualModel instance."""
    asset = asset.upper()
    x = x_nodes[:, np.newaxis]   # (n, 1)
    y = y_nodes[np.newaxis, :]   # (1, m)

    prices = np.empty(len(strikes))
    for i, K in enumerate(strikes):
        if asset == "X":
            payoff = np.maximum(x - K, 0.0) * np.ones_like(W_ij)
        elif asset == "Y":
            payoff = np.maximum(y - K, 0.0) * np.ones_like(W_ij)
        elif asset == "Z":
            payoff = np.maximum(x - y - K, 0.0)
        else:
            raise ValueError(f"asset must be 'X', 'Y', or 'Z', got '{asset}'.")
        prices[i] = float(np.sum(W_ij * payoff))
    return prices


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_calibration_diagnostics(W_ij:        np.ndarray,
                                    x_nodes:     np.ndarray,
                                    y_nodes:     np.ndarray,
                                    market_data: Dict,
                                    epsilon:     float) -> Dict:
    """Compute comprehensive calibration quality diagnostics.

    Parameters
    ----------
    W_ij : ndarray (n, m)
        Model joint probability weights (should sum ≈ 1).
    x_nodes : ndarray (n,)
        Quadrature nodes for X.
    y_nodes : ndarray (m,)
        Quadrature nodes for Y.
    market_data : dict
        Output of :meth:`SyntheticCMSMarket.get_market_data`.
    epsilon : float
        Entropic regularisation parameter (informational).

    Returns
    -------
    dict
        ``normalization_error``, ``X_moment_error``, ``Y_moment_error``,
        ``X_price_errors``, ``Y_price_errors``, ``Z_price_errors``,
        ``X_vol_errors``, ``Y_vol_errors``, ``Z_vol_errors``  (all in bps),
        ``max_abs_price_error``, ``max_abs_vol_error_bps``.
    """
    F_X = market_data["F_X"]
    F_Y = market_data["F_Y"]
    F_Z = market_data["F_Z"]
    T   = market_data["T"]
    K_X, C_X = market_data["K_X"], market_data["C_X"]
    K_Y, C_Y = market_data["K_Y"], market_data["C_Y"]
    K_Z, C_Z = market_data["K_Z"], market_data["C_Z"]

    x = x_nodes[:, np.newaxis]
    y = y_nodes[np.newaxis, :]

    # Normalisation
    norm_err = abs(1.0 - float(W_ij.sum()))

    # Moment errors
    E_X_model = float(np.sum(W_ij * x))
    E_Y_model = float(np.sum(W_ij * y))
    x_mom_err = abs(F_X - E_X_model)
    y_mom_err = abs(F_Y - E_Y_model)

    # Model call prices
    C_X_model = _model_call_prices(W_ij, x_nodes, y_nodes, K_X, "X")
    C_Y_model = _model_call_prices(W_ij, x_nodes, y_nodes, K_Y, "Y")
    C_Z_model = _model_call_prices(W_ij, x_nodes, y_nodes, K_Z, "Z")

    # Price errors
    X_price_err = C_X_model - C_X
    Y_price_err = C_Y_model - C_Y
    Z_price_err = C_Z_model - C_Z

    # Implied-vol errors (in bps) — skip deep-OTM strikes where prices are
    # near-zero and Bachelier implied-vol inversion is ill-conditioned.
    # A price below 0.5 bps * min(sigma_atm)*sqrt(T) is treated as zero.
    min_price_threshold = 1e-7   # ~0.01 bps in rate space; below this skip vol conv.

    def _safe_vol_err(C_mkt: np.ndarray, C_mdl: np.ndarray,
                      F: float, K: np.ndarray) -> np.ndarray:
        """Return vol error in bps, NaN where either price is below threshold."""
        liquid = (C_mkt > min_price_threshold) & (C_mdl > min_price_threshold)
        v_mkt  = prices_to_bachelier_vols(np.where(liquid, C_mkt, np.nan), F, K, T)
        v_mdl  = prices_to_bachelier_vols(np.where(liquid, C_mdl, np.nan), F, K, T)
        err    = (v_mdl - v_mkt) / _BPS
        err[~liquid] = np.nan
        return err

    X_vol_err = _safe_vol_err(C_X, C_X_model, F_X, K_X)
    Y_vol_err = _safe_vol_err(C_Y, C_Y_model, F_Y, K_Y)
    Z_vol_err = _safe_vol_err(C_Z, C_Z_model, F_Z, K_Z)

    all_price_errs = np.concatenate([
        np.abs(X_price_err), np.abs(Y_price_err), np.abs(Z_price_err)
    ])
    all_vol_errs = np.concatenate([
        np.abs(X_vol_err[np.isfinite(X_vol_err)]),
        np.abs(Y_vol_err[np.isfinite(Y_vol_err)]),
        np.abs(Z_vol_err[np.isfinite(Z_vol_err)]),
    ])

    return {
        "normalization_error":  norm_err,
        "X_moment_error":       x_mom_err,
        "Y_moment_error":       y_mom_err,
        "X_price_errors":       X_price_err,
        "Y_price_errors":       Y_price_err,
        "Z_price_errors":       Z_price_err,
        "X_vol_errors":         X_vol_err,
        "Y_vol_errors":         Y_vol_err,
        "Z_vol_errors":         Z_vol_err,
        "max_abs_price_error":  float(all_price_errs.max()) if len(all_price_errs) else np.nan,
        "max_abs_vol_error_bps": float(all_vol_errs.max()) if len(all_vol_errs) else np.nan,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_smile_comparison(market_data:  Dict,
                          W_ij:         np.ndarray,
                          x_nodes:      np.ndarray,
                          y_nodes:      np.ndarray,
                          title_suffix: str = "",
                          save_path:    Optional[str] = None) -> plt.Figure:
    """Three-panel implied-vol smile comparison: market vs model.

    Parameters
    ----------
    market_data : dict
        Output of :meth:`SyntheticCMSMarket.get_market_data`.
    W_ij : ndarray (n, m)
        Model joint probability weights.
    x_nodes, y_nodes : ndarray
        Quadrature nodes.
    title_suffix : str
        Appended to each sub-panel title.
    save_path : str or None
        If provided, the figure is saved to this path and closed.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    T   = market_data["T"]
    F_X, F_Y, F_Z = market_data["F_X"], market_data["F_Y"], market_data["F_Z"]
    K_X, C_X = market_data["K_X"], market_data["C_X"]
    K_Y, C_Y = market_data["K_Y"], market_data["C_Y"]
    K_Z, C_Z = market_data["K_Z"], market_data["C_Z"]

    C_X_model = _model_call_prices(W_ij, x_nodes, y_nodes, K_X, "X")
    C_Y_model = _model_call_prices(W_ij, x_nodes, y_nodes, K_Y, "Y")
    C_Z_model = _model_call_prices(W_ij, x_nodes, y_nodes, K_Z, "Z")

    def to_bps_vol(prices, F, K):
        vols = prices_to_bachelier_vols(prices, F, K, T)
        return vols / _BPS   # display in bps

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (K, C_mkt, C_mdl, F, label) in zip(axes, [
        (K_X, C_X, C_X_model, F_X, "X (CMS long leg)"),
        (K_Y, C_Y, C_Y_model, F_Y, "Y (CMS short leg)"),
        (K_Z, C_Z, C_Z_model, F_Z, "Z = X − Y (spread)"),
    ]):
        mkt_bps = to_bps_vol(C_mkt, F, K)
        mdl_bps = to_bps_vol(C_mdl, F, K)

        K_bp = (K - F) / _BPS    # x-axis: moneyness in bps

        valid = np.isfinite(mkt_bps) & np.isfinite(mdl_bps)
        ax.plot(K_bp[valid], mkt_bps[valid], "o-",
                color="steelblue", linewidth=1.8, markersize=4, label="Market")
        ax.plot(K_bp[valid], mdl_bps[valid], "s--",
                color="firebrick", linewidth=1.8, markersize=4, label="Model")
        ax.set_xlabel("Moneyness (K − F)  [bps]")
        ax.set_ylabel("Bachelier vol  [bps]")
        ax.set_title(f"{label}{(' — ' + title_suffix) if title_suffix else ''}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_joint_distribution(W_ij:      np.ndarray,
                            x_nodes:   np.ndarray,
                            y_nodes:   np.ndarray,
                            save_path: Optional[str] = None) -> plt.Figure:
    """Heatmap of the calibrated joint distribution W_ij.

    Parameters
    ----------
    W_ij : ndarray (n, m)
    x_nodes : ndarray (n,)
        Quadrature nodes for X (row axis).
    y_nodes : ndarray (m,)
        Quadrature nodes for Y (column axis).
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Convert rates to bps for display
    x_bps = x_nodes / _BPS
    y_bps = y_nodes / _BPS

    im = ax.pcolormesh(
        y_bps, x_bps, W_ij,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="Joint probability weight $W_{ij}$")
    ax.set_xlabel("Y (CMS short leg)  [bps]")
    ax.set_ylabel("X (CMS long leg)  [bps]")
    ax.set_title("Calibrated joint distribution")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Implied Gaussian correlation
# ---------------------------------------------------------------------------

def gaussian_copula_price(marginal_X:   MarginalDistribution,
                          marginal_Y:   MarginalDistribution,
                          rho_impl:     float,
                          strike:       float,
                          quad_grid:    Dict) -> float:
    """Price  max(X + Y − K, 0)  under the Gaussian copula with *rho_impl*.

    Builds the discrete joint measure from the copula + marginals on the
    existing quadrature grid and computes the call payoff expectation.

    Parameters
    ----------
    marginal_X, marginal_Y : MarginalDistribution
    rho_impl : float
        Gaussian copula correlation.
    strike : float
        Call option strike on  X + Y.
    quad_grid : dict
        Output of :func:`~src.quadrature.build_quadrature_grid`.

    Returns
    -------
    float
        Option price.
    """
    Q_ij = build_prior_weights(
        marginal_X, marginal_Y, quad_grid,
        copula_type="gaussian", rho=rho_impl,
    )
    # Normalise (should already sum to 1 after build_prior_weights)
    Q_ij = Q_ij / Q_ij.sum()

    x = quad_grid["x_nodes"][:, np.newaxis]
    y = quad_grid["y_nodes"][np.newaxis, :]
    payoff = np.maximum(x + y - strike, 0.0)
    return float(np.sum(Q_ij * payoff))


def compute_implied_gaussian_correlation(exotic_price: float,
                                         marginal_X:   MarginalDistribution,
                                         marginal_Y:   MarginalDistribution,
                                         strike:       float,
                                         quad_grid:    Dict,
                                         rho_bounds:   Tuple[float, float] = (-0.999, 0.999)
                                         ) -> float:
    """Find the Gaussian copula correlation that reproduces *exotic_price*.

    Inverts :func:`gaussian_copula_price` via Brent's method.

    Parameters
    ----------
    exotic_price : float
        Target option price  E[max(X + Y − K, 0)].
    marginal_X, marginal_Y : MarginalDistribution
    strike : float
    quad_grid : dict
    rho_bounds : (float, float)
        Search interval for rho (default (-0.999, 0.999)).

    Returns
    -------
    rho_impl : float
        Implied Gaussian correlation.

    Raises
    ------
    ValueError
        If the price cannot be matched within *rho_bounds*.
    """
    def objective(rho: float) -> float:
        return gaussian_copula_price(
            marginal_X, marginal_Y, rho, strike, quad_grid
        ) - exotic_price

    rho_lo, rho_hi = rho_bounds
    f_lo = objective(rho_lo)
    f_hi = objective(rho_hi)

    if f_lo * f_hi > 0:
        raise ValueError(
            f"Cannot bracket the implied correlation: "
            f"price {exotic_price:.6g} may be outside the range achievable "
            f"by the Gaussian copula on rho in [{rho_lo}, {rho_hi}]. "
            f"f(rho_lo)={f_lo:.4g}, f(rho_hi)={f_hi:.4g}."
        )

    rho_impl = brentq(objective, rho_lo, rho_hi, xtol=1e-8, rtol=1e-8)
    return float(rho_impl)
