"""
Experiment 4 – Prior sensitivity analysis.

Compares calibrated exotic prices under four different copula priors with
fixed correlation rho = 0.78:

  1. Gaussian copula
  2. Student-t copula, ν = 1  (heavy tails)
  3. Student-t copula, ν = 3
  4. Student-t copula, ν = 10  (near-Gaussian)

Setup
-----
* Market  : 2s20s, T = 5Y, synthetic data
* Grid    : n = 100 Gauss-Legendre nodes
* Model   : epsilon = 1, beta = 0 (baseline, no exotic tilt)
* Output  : results/exp4_results.csv, results/exp4_prior_sensitivity.png
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.market_data_loader import load_cms_market_data
from src.marginals import MarginalDistribution
from src.quadrature import build_quadrature_grid
from src.copulas import build_prior_weights
from src.dual_model import DualModel
from src.reporting import gaussian_copula_price, compute_implied_gaussian_correlation


# Prior specifications: (label, copula_type, nu)
_PRIORS = [
    ("Gaussian",     "gaussian", None),
    ("t-copula ν=1", "t",        1.0),
    ("t-copula ν=3", "t",        3.0),
    ("t-copula ν=10","t",        10.0),
]


def run_exp4(save_results: bool = True) -> pd.DataFrame:
    """Run prior sensitivity analysis and return results DataFrame.

    Parameters
    ----------
    save_results : bool

    Returns
    -------
    pd.DataFrame
    """
    print("=" * 60)
    print("Experiment 4: Prior sensitivity (rho=0.78, ε=1, β=0)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Shared market setup
    # ------------------------------------------------------------------
    market_data = load_cms_market_data(scenario="2s20s", expiry_years=5.0)

    margX = MarginalDistribution(
        strikes=market_data["K_X"], call_prices=market_data["C_X"],
        forward=market_data["F_X"], T=market_data["T"],
    )
    margY = MarginalDistribution(
        strikes=market_data["K_Y"], call_prices=market_data["C_Y"],
        forward=market_data["F_Y"], T=market_data["T"],
    )

    grid = build_quadrature_grid(margX, margY, n=100)

    x = grid["x_nodes"]
    y = grid["y_nodes"]

    # ------------------------------------------------------------------
    # Exotic strike grid (same as exp3, derived from Gaussian prior)
    # ------------------------------------------------------------------
    Q_ref = build_prior_weights(
        margX, margY, grid, copula_type="gaussian", rho=0.78
    )
    mu_sum = market_data["F_X"] + market_data["F_Y"]
    Q_sum2 = np.sum(Q_ref * (x[:, np.newaxis] + y[np.newaxis, :]) ** 2)
    std_sum = float(np.sqrt(max(Q_sum2 - mu_sum ** 2, 1e-8)))
    K_exotic = np.linspace(mu_sum - 2.0 * std_sum, mu_sum + 2.0 * std_sum, 7)
    print(f"  K_exotic: {K_exotic}")

    rows = []

    for prior_label, copula_type, nu in _PRIORS:
        print(f"\n  Prior: {prior_label}")

        # Build copula-specific Q_ij
        Q_ij = build_prior_weights(
            margX, margY, grid,
            copula_type=copula_type, rho=0.78,
            **({"nu": nu} if nu is not None else {}),
        )

        for K in K_exotic:
            exotic_payoff = np.maximum(
                x[:, np.newaxis] + y[np.newaxis, :] - K, 0.0
            )
            model = DualModel(
                x_nodes=grid["x_nodes"],
                y_nodes=grid["y_nodes"],
                x_weights=grid["x_weights"],
                y_weights=grid["y_weights"],
                Q_ij=Q_ij,
                K_X=market_data["K_X"],
                K_Y=market_data["K_Y"],
                K_Z=market_data["K_Z"],
                C_X_prices=market_data["C_X"],
                C_Y_prices=market_data["C_Y"],
                C_Z_prices=market_data["C_Z"],
                E_X=market_data["F_X"],
                E_Y=market_data["F_Y"],
                epsilon=1.0,
                beta=0.0,
            )
            result = model.solve(tol=1e-5, max_iter=100)
            W_ij = result["W_ij"]
            exotic_price = model.compute_exotic_price(W_ij, exotic_payoff)

            try:
                rho_impl = compute_implied_gaussian_correlation(
                    exotic_price, margX, margY, K, grid
                )
            except ValueError:
                rho_impl = np.nan

            print(f"    K={K:.4f}  price={exotic_price:.6f}  "
                  f"rho_impl={rho_impl:.4f}" if not np.isnan(rho_impl)
                  else f"    K={K:.4f}  price={exotic_price:.6f}  rho_impl=NaN")

            rows.append(dict(
                prior=prior_label,
                copula_type=copula_type,
                nu=nu if nu is not None else np.inf,
                K=float(K),
                exotic_price=float(exotic_price),
                rho_impl=float(rho_impl),
            ))

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plot: implied correlation vs strike for each prior
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = ["-o", "--s", ":^", "-.D"]

    for (prior_label, _, _), style in zip(_PRIORS, styles):
        sub = df[df["prior"] == prior_label].sort_values("K")
        ax.plot(sub["K"], sub["rho_impl"], style,
                label=prior_label, linewidth=1.8, markersize=5)

    ax.axhline(0.78, color="gray", linestyle=":", linewidth=1,
               label="Prior rho=0.78")
    ax.set_xlabel("Strike K (call on X+Y)")
    ax.set_ylabel("Implied Gaussian correlation ρ")
    ax.set_title("Experiment 4 – Prior sensitivity: implied correlation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_results:
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/exp4_results.csv", index=False)
        fig.savefig("results/exp4_prior_sensitivity.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("\n  Saved: results/exp4_results.csv, results/exp4_prior_sensitivity.png")

    return df


if __name__ == "__main__":
    run_exp4()
