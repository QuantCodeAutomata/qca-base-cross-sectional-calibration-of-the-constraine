"""
Experiment 3 – Epsilon sweep for no-arbitrage bounds on call(X + Y).

Sign convention (from the paper):
  β = −1  maximises the exotic payoff → **upper** bound on the price.
  β = +1  minimises the exotic payoff → **lower** bound on the price.
(The term −β·C_ij appears inside A_ij, so negative β boosts W_ij where
C_ij is large, pushing mass toward high-payoff scenarios.)

Setup
-----
* Market   : 2s20s, T = 5Y, synthetic data
* Grid     : n = 100 Gauss-Legendre nodes  (smaller for multi-solve speed)
* Prior    : Gaussian copula, rho = 0.78
* Epsilons : [1.0, 0.01, 0.001]
* Betas    : {+1, -1}
* Strikes  : 7 strikes spanning ±2 std(X+Y) around E[X+Y]
* Output   : results/exp3_results.csv, results/exp3_bounds.png
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


def run_exp3(save_results: bool = True) -> pd.DataFrame:
    """Run epsilon sweep and return results DataFrame.

    Parameters
    ----------
    save_results : bool
        Save CSV and bounds plot to ``results/`` when *True*.

    Returns
    -------
    pd.DataFrame
    """
    print("=" * 60)
    print("Experiment 3: Epsilon sweep for no-arbitrage bounds")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Setup
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
    Q_ij = build_prior_weights(
        margX, margY, grid, copula_type="gaussian", rho=0.78
    )

    x = grid["x_nodes"]
    y = grid["y_nodes"]

    # ------------------------------------------------------------------
    # Exotic strike grid: 7 strikes around E[X+Y] ± 2 std(X+Y)
    # ------------------------------------------------------------------
    mu_sum = market_data["F_X"] + market_data["F_Y"]

    # Approximate std(X+Y) from Q prior
    Q_sum = np.sum(Q_ij * (x[:, np.newaxis] + y[np.newaxis, :]) ** 2)
    std_sum = float(np.sqrt(max(Q_sum - mu_sum ** 2, 1e-8)))

    K_exotic = np.linspace(mu_sum - 2.0 * std_sum, mu_sum + 2.0 * std_sum, 7)
    print(f"  E[X+Y]={mu_sum:.4f}, std≈{std_sum:.4f}")
    print(f"  K_exotic: {K_exotic}")

    epsilon_values = [1.0, 0.01, 0.001]
    beta_values = [+1.0, -1.0]

    rows = []

    for eps in epsilon_values:
        for beta in beta_values:
            # β=−1 maximises the exotic (upper); β=+1 minimises (lower).
            beta_label = "upper" if beta < 0 else "lower"
            print(f"\n  ε={eps}, β={beta:+.0f} ({beta_label})")

            theta_warm = None
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
                    epsilon=eps,
                    beta=float(beta),
                    exotic_payoff_matrix=exotic_payoff,
                )
                result = model.solve(
                    theta_init=theta_warm, tol=1e-5, max_iter=100
                )
                theta_warm = result["theta"]
                W_ij = result["W_ij"]
                exotic_price = model.compute_exotic_price(W_ij, exotic_payoff)

                # Implied Gaussian correlation
                try:
                    rho_impl = compute_implied_gaussian_correlation(
                        exotic_price, margX, margY, K, grid
                    )
                except ValueError:
                    rho_impl = np.nan

                print(f"    K={K:.4f}  price={exotic_price:.6f}  "
                      f"rho_impl={rho_impl:.4f}" if not np.isnan(rho_impl)
                      else f"    K={K:.4f}  price={exotic_price:.6f}  "
                           f"rho_impl=NaN")

                rows.append(dict(
                    epsilon=eps,
                    beta=float(beta),
                    bound=beta_label,
                    K=float(K),
                    exotic_price=float(exotic_price),
                    rho_impl=float(rho_impl),
                ))

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plot: implied correlation bounds vs strike for each epsilon
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(epsilon_values),
                             figsize=(5 * len(epsilon_values), 5),
                             sharey=True)

    for ax, eps in zip(axes, epsilon_values):
        sub = df[df["epsilon"] == eps]
        upper = sub[sub["bound"] == "upper"].sort_values("K")
        lower = sub[sub["bound"] == "lower"].sort_values("K")

        ax.plot(upper["K"], upper["rho_impl"], "o-r",
                label="Upper (β=−1)", linewidth=1.8, markersize=5)
        ax.plot(lower["K"], lower["rho_impl"], "s--b",
                label="Lower (β=+1)", linewidth=1.8, markersize=5)
        ax.axhline(0.78, color="gray", linestyle=":", linewidth=1,
                   label="Prior rho=0.78")
        ax.set_xlabel("Strike K (call on X+Y)")
        ax.set_ylabel("Implied Gaussian correlation ρ")
        ax.set_title(f"ε = {eps}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 3 – No-arbitrage correlation bounds", fontsize=13)
    fig.tight_layout()

    if save_results:
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/exp3_results.csv", index=False)
        fig.savefig("results/exp3_bounds.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("\n  Saved: results/exp3_results.csv, results/exp3_bounds.png")

    return df


if __name__ == "__main__":
    run_exp3()
