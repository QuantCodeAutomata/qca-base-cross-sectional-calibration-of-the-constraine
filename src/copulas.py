"""Copula density functions and prior-weight construction.

Provides:
- Gaussian copula density (bivariate normal with correlation *rho*)
- Student-t copula density (bivariate t with correlation *rho* and *nu* d.o.f.)
- ``build_prior_weights``: assembles the discrete prior Q_ij on the
  quadrature grid from copula + marginal densities.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm, t as student_t

from .marginals import MarginalDistribution


# ---------------------------------------------------------------------------
# Copula densities
# ---------------------------------------------------------------------------

def gaussian_copula_density(u: float | np.ndarray,
                             v: float | np.ndarray,
                             rho: float) -> float | np.ndarray:
    """Bivariate Gaussian copula density  c(u, v; rho).

    Parameters
    ----------
    u, v : float or array
        Uniform margins in (0, 1).
    rho : float
        Linear correlation parameter in (-1, 1).

    Returns
    -------
    float or ndarray
        Copula density value(s).
    """
    u = np.clip(np.asarray(u, dtype=float), 1e-12, 1.0 - 1e-12)
    v = np.clip(np.asarray(v, dtype=float), 1e-12, 1.0 - 1e-12)

    z1 = norm.ppf(u)
    z2 = norm.ppf(v)

    rho2 = rho * rho
    log_c = (
        -0.5 * np.log(1.0 - rho2)
        - (rho2 * (z1 ** 2 + z2 ** 2) - 2.0 * rho * z1 * z2)
        / (2.0 * (1.0 - rho2))
    )
    return np.exp(log_c)


def t_copula_density(u: float | np.ndarray,
                     v: float | np.ndarray,
                     rho: float,
                     nu: float) -> float | np.ndarray:
    """Bivariate Student-t copula density  c(u, v; rho, nu).

    Parameters
    ----------
    u, v : float or array
        Uniform margins in (0, 1).
    rho : float
        Correlation parameter in (-1, 1).
    nu : float
        Degrees of freedom (> 0).

    Returns
    -------
    float or ndarray
        Copula density value(s).
    """
    u = np.clip(np.asarray(u, dtype=float), 1e-12, 1.0 - 1e-12)
    v = np.clip(np.asarray(v, dtype=float), 1e-12, 1.0 - 1e-12)

    t1 = student_t.ppf(u, df=nu)
    t2 = student_t.ppf(v, df=nu)

    rho2  = rho * rho
    q_rho = (t1 ** 2 - 2.0 * rho * t1 * t2 + t2 ** 2) / (1.0 - rho2)

    log_c = (
        gammaln((nu + 2.0) / 2.0)
        + gammaln(nu / 2.0)
        - 2.0 * gammaln((nu + 1.0) / 2.0)
        - 0.5 * np.log(1.0 - rho2)
        - ((nu + 2.0) / 2.0) * np.log(1.0 + q_rho / nu)
        + ((nu + 1.0) / 2.0) * (np.log(1.0 + t1 ** 2 / nu)
                                  + np.log(1.0 + t2 ** 2 / nu))
    )
    return np.exp(log_c)


# ---------------------------------------------------------------------------
# Prior weight construction
# ---------------------------------------------------------------------------

def build_prior_weights(marginal_X: MarginalDistribution,
                        marginal_Y: MarginalDistribution,
                        quad_grid: dict,
                        copula_type: str = "gaussian",
                        rho: float = 0.78,
                        nu: float | None = None) -> np.ndarray:
    """Assemble the discrete prior  Q_ij  on the quadrature grid.

    The prior is a discretised joint density constructed from a copula
    and the two marginals:

        Q_ij  ∝  c(F_X(x_i), F_Y(y_j))  f_X(x_i)  f_Y(y_j)  wx_i  wy_j

    followed by clipping to [0, ∞) and normalisation to sum 1.

    Parameters
    ----------
    marginal_X, marginal_Y : MarginalDistribution
        Fitted marginal distributions.
    quad_grid : dict
        Output of :func:`~src.quadrature.build_quadrature_grid`.
    copula_type : {'gaussian', 't'}
        Which copula family to use.
    rho : float
        Correlation parameter (default 0.78).
    nu : float or None
        Degrees of freedom for the t-copula (required when
        ``copula_type='t'``).

    Returns
    -------
    Q_ij : ndarray, shape (n, m)
        Normalised prior weights, sum = 1.
    """
    if copula_type == "t" and nu is None:
        raise ValueError("nu must be provided for the t-copula.")

    x_nodes   = quad_grid["x_nodes"]    # shape (n,)
    y_nodes   = quad_grid["y_nodes"]    # shape (m,)
    x_weights = quad_grid["x_weights"]  # shape (n,)
    y_weights = quad_grid["y_weights"]  # shape (m,)

    n, m = len(x_nodes), len(y_nodes)

    # ---- marginal CDF / PDF on the quadrature nodes --------------------
    # Vectorised evaluation
    u_vec = np.clip(marginal_X.cdf(x_nodes), 1e-12, 1.0 - 1e-12)  # (n,)
    v_vec = np.clip(marginal_Y.cdf(y_nodes), 1e-12, 1.0 - 1e-12)  # (m,)

    fx_vec = np.clip(marginal_X.pdf(x_nodes), 0.0, None)           # (n,)
    fy_vec = np.clip(marginal_Y.pdf(y_nodes), 0.0, None)           # (m,)

    # ---- copula density on the (n × m) grid ---------------------------
    # Broadcast to (n, m)
    U = u_vec[:, np.newaxis]   # (n, 1)
    V = v_vec[np.newaxis, :]   # (1, m)

    if copula_type == "gaussian":
        C_ij = gaussian_copula_density(U, V, rho)
    elif copula_type == "t":
        C_ij = t_copula_density(U, V, rho, nu)
    else:
        raise ValueError(
            f"Unknown copula_type '{copula_type}'. Choose 'gaussian' or 't'."
        )

    # ---- joint prior weights ------------------------------------------
    # f_X(x_i) broadcast over j, f_Y(y_j) broadcast over i
    FX = fx_vec[:, np.newaxis]   # (n, 1)
    FY = fy_vec[np.newaxis, :]   # (1, m)
    WX = x_weights[:, np.newaxis]  # (n, 1)
    WY = y_weights[np.newaxis, :]  # (1, m)

    Q_ij = C_ij * FX * FY * WX * WY   # (n, m)

    # ---- clip and normalise -------------------------------------------
    Q_ij = np.clip(Q_ij, 0.0, None)
    total = Q_ij.sum()
    if total <= 0.0:
        raise RuntimeError(
            "Prior weight matrix sums to zero. "
            "Check marginals, copula parameters, and quadrature grid."
        )
    Q_ij /= total
    return Q_ij
