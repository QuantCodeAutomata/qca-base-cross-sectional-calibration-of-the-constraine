"""
Experiment 5 – High-spread-vol stress scenario.

Repeats the epsilon sweep (Experiment 3) under the stressed 30y2y
market scenario and compares bound widths with the 2s20s base case.

Setup
-----
* Market   : 30y2y stressed, T = 5Y
* Grid     : n = 100 Gauss-Legendre nodes
* Prior    : Gaussian copula, rho = 0.78
* Epsilons : [1.0, 0.01, 0.001]
* Betas    : {+1, −1}
* Strikes  : 7 strikes spanning ±2 std(X+Y) around E[X+Y]
* Output   : results/exp5_results.csv, results/exp5_stress_bounds.png
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


def _run_sweep(market_data: dict,
               grid: dict,
               margX: object,
               margY: object,
               Q_ij: np.ndarray,
               epsilon_values: list,
               K_exotic: np.ndarray,
               label: str) -> list:
    """Run the epsilon sweep and return a list of result dicts."""
    x, y = grid["x_nodes"], grid["y_nodes"]
    rows = []

    for eps in epsilon_values:
        for beta in [+1.0, -1.0]:
            # β=−1 maximises the exotic (upper); β=+1 minimises (lower).
            bound = "upper" if beta < 0 else "lower"
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

                try:
                    rho_impl = compute_implied_gaussian_correlation(
                        exotic_price, margX, margY, K, grid
                    )
                except (ValueError, RuntimeError):
                    rho_impl = np.nan

                rows.append(dict(
                    scenario=label,
                    epsilon=eps,
                    beta=float(beta),
                    bound=bound,
                    K=float(K),
                    exotic_price=float(exotic_price),
                    rho_impl=float(rho_impl),
                ))
    return rows


def run_exp5(save_results: bool = True) -> pd.DataFrame:
    """Run stress-scenario comparison and return results DataFrame.

    Parameters
    ----------
    save_results : bool

    Returns
    -------
    pd.DataFrame
    """
    print("=" * 60)
    print("Experiment 5: High-spread-vol stress scenario (30y2y)")
    print("=" * 60)

    epsilon_values = [1.0, 0.01, 0.001]
    all_rows = []

    # ------------------------------------------------------------------
    # Base case: 2s20s (for comparison)
    # ------------------------------------------------------------------
    print("\n  [Base case: 2s20s]")
    data_base = load_cms_market_data(scenario="2s20s", expiry_years=5.0)
    mX_base = MarginalDistribution(
        strikes=data_base["K_X"], call_prices=data_base["C_X"],
        forward=data_base["F_X"], T=data_base["T"],
    )
    mY_base = MarginalDistribution(
        strikes=data_base["K_Y"], call_prices=data_base["C_Y"],
        forward=data_base["F_Y"], T=data_base["T"],
    )
    grid_base = build_quadrature_grid(mX_base, mY_base, n=100)
    Q_base = build_prior_weights(
        mX_base, mY_base, grid_base, copula_type="gaussian", rho=0.78
    )
    x_b, y_b = grid_base["x_nodes"], grid_base["y_nodes"]
    mu_b = data_base["F_X"] + data_base["F_Y"]
    Q_sum2_b = np.sum(Q_base * (x_b[:, None] + y_b[None, :]) ** 2)
    std_b = float(np.sqrt(max(Q_sum2_b - mu_b ** 2, 1e-8)))
    K_base = np.linspace(mu_b - 2 * std_b, mu_b + 2 * std_b, 7)

    rows_base = _run_sweep(
        data_base, grid_base, mX_base, mY_base, Q_base,
        epsilon_values, K_base, label="2s20s-base"
    )
    all_rows.extend(rows_base)

    # ------------------------------------------------------------------
    # Stress scenario: 30y2y stressed
    # ------------------------------------------------------------------
    print("\n  [Stress scenario: 30y2y stressed]")
    data_stress = load_cms_market_data(
        scenario="30y2y", expiry_years=5.0, stressed=True
    )
    print(f"  F_X={data_stress['F_X']:.4f}, F_Y={data_stress['F_Y']:.4f}, "
          f"sigma_X={data_stress['sigma_X']:.4f}, sigma_Y={data_stress['sigma_Y']:.4f}")

    mX_stress = MarginalDistribution(
        strikes=data_stress["K_X"], call_prices=data_stress["C_X"],
        forward=data_stress["F_X"], T=data_stress["T"],
    )
    mY_stress = MarginalDistribution(
        strikes=data_stress["K_Y"], call_prices=data_stress["C_Y"],
        forward=data_stress["F_Y"], T=data_stress["T"],
    )
    grid_stress = build_quadrature_grid(mX_stress, mY_stress, n=100)
    Q_stress = build_prior_weights(
        mX_stress, mY_stress, grid_stress, copula_type="gaussian", rho=0.78
    )
    x_s, y_s = grid_stress["x_nodes"], grid_stress["y_nodes"]
    mu_s = data_stress["F_X"] + data_stress["F_Y"]
    Q_sum2_s = np.sum(Q_stress * (x_s[:, None] + y_s[None, :]) ** 2)
    std_s = float(np.sqrt(max(Q_sum2_s - mu_s ** 2, 1e-8)))
    K_stress = np.linspace(mu_s - 2 * std_s, mu_s + 2 * std_s, 7)

    rows_stress = _run_sweep(
        data_stress, grid_stress, mX_stress, mY_stress, Q_stress,
        epsilon_values, K_stress, label="30y2y-stressed"
    )
    all_rows.extend(rows_stress)

    df = pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # Compare bound widths
    # ------------------------------------------------------------------
    print("\n  Bound widths (rho_upper - rho_lower) at smallest ε:")
    for scenario in ["2s20s-base", "30y2y-stressed"]:
        sub = df[(df["scenario"] == scenario) & (df["epsilon"] == 0.001)]
        if sub.empty:
            continue
        upper = sub[sub["bound"] == "upper"]
        lower = sub[sub["bound"] == "lower"]
        if upper.empty or lower.empty:
            continue
        width = float(
            upper["rho_impl"].mean() - lower["rho_impl"].mean()
        )
        print(f"    {scenario}: mean width = {width:.4f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharey=False
    )
    scenario_titles = {
        "2s20s-base": "2s20s (base)",
        "30y2y-stressed": "30y2y (stressed)",
    }

    for ax, (scenario_key, ax_title) in zip(
        axes, scenario_titles.items()
    ):
        sub_sc = df[df["scenario"] == scenario_key]
        colors = plt.cm.tab10(np.linspace(0, 0.5, len(epsilon_values)))

        for eps, col in zip(epsilon_values, colors):
            sub_eps = sub_sc[sub_sc["epsilon"] == eps]
            upper = sub_eps[sub_eps["bound"] == "upper"].sort_values("K")
            lower = sub_eps[sub_eps["bound"] == "lower"].sort_values("K")

            if not upper.empty and not lower.empty:
                ax.plot(upper["K"], upper["rho_impl"], "-",
                        color=col, linewidth=1.8,
                        label=f"ε={eps} upper (β=−1)")
                ax.plot(lower["K"], lower["rho_impl"], "--",
                        color=col, linewidth=1.8,
                        label=f"ε={eps} lower (β=+1)")
                # fill only where both bounds are finite
                rho_up = upper["rho_impl"].values
                rho_lo = lower["rho_impl"].values
                finite = np.isfinite(rho_up) & np.isfinite(rho_lo)
                if finite.any():
                    ax.fill_between(
                        upper["K"].values[finite],
                        rho_lo[finite], rho_up[finite],
                        alpha=0.15, color=col
                    )

        ax.axhline(0.78, color="gray", linestyle=":", linewidth=1,
                   label="Prior rho=0.78")
        ax.set_xlabel("Strike K (call on X+Y)")
        ax.set_ylabel("Implied Gaussian correlation ρ")
        ax.set_title(ax_title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 5 – Stress scenario bounds comparison", fontsize=13)
    fig.tight_layout()

    if save_results:
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/exp5_results.csv", index=False)
        fig.savefig("results/exp5_stress_bounds.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("\n  Saved: results/exp5_results.csv, results/exp5_stress_bounds.png")

    return df


if __name__ == "__main__":
    run_exp5()
