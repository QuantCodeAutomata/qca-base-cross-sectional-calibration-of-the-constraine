"""
Experiment 1 – Base Schrödinger OT calibration.

Setup
-----
* Market  : 2s20s, T = 5Y, synthetic data
* Grid    : n = 200 Gauss-Legendre nodes
* Prior   : Gaussian copula, rho = 0.78
* Model   : epsilon = 1, beta = 0 (no exotic tilt)
* Output  : calibration diagnostics, smile plots, joint-distribution heatmap
"""
from __future__ import annotations

import json
import os

import numpy as np

from data.market_data_loader import load_cms_market_data
from src.marginals import MarginalDistribution
from src.quadrature import build_quadrature_grid
from src.copulas import build_prior_weights
from src.dual_model import DualModel
from src.reporting import (
    compute_calibration_diagnostics,
    plot_smile_comparison,
    plot_joint_distribution,
)


def run_exp1(save_results: bool = True) -> dict:
    """Run base calibration experiment and return a results dict.

    Parameters
    ----------
    save_results : bool
        When *True*, saves plots and metrics to ``results/``.

    Returns
    -------
    dict
        Keys: scenario, epsilon, beta, converged, iterations,
        final_grad_norm, normalization_error, max_abs_price_error,
        max_abs_vol_error_bps.
    """
    print("=" * 60)
    print("Experiment 1: Base calibration  (ε=1, β=0, 2s20s)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load market data
    # ------------------------------------------------------------------
    market_data = load_cms_market_data(scenario="2s20s", expiry_years=5.0)
    print(f"  Market: F_X={market_data['F_X']:.4f}, "
          f"F_Y={market_data['F_Y']:.4f}, F_Z={market_data['F_Z']:.4f}")

    # ------------------------------------------------------------------
    # 2. Build marginal distributions M^X and M^Y
    # ------------------------------------------------------------------
    margX = MarginalDistribution(
        strikes=market_data["K_X"],
        call_prices=market_data["C_X"],
        forward=market_data["F_X"],
        T=market_data["T"],
    )
    margY = MarginalDistribution(
        strikes=market_data["K_Y"],
        call_prices=market_data["C_Y"],
        forward=market_data["F_Y"],
        T=market_data["T"],
    )

    # ------------------------------------------------------------------
    # 3. Build 200-point GL quadrature grid
    # ------------------------------------------------------------------
    grid = build_quadrature_grid(margX, margY, n=200)
    print(f"  Grid: n={len(grid['x_nodes'])} nodes, "
          f"x∈[{grid['x_low']:.4f}, {grid['x_high']:.4f}]  "
          f"y∈[{grid['y_low']:.4f}, {grid['y_high']:.4f}]")

    # ------------------------------------------------------------------
    # 4. Build Gaussian copula prior Q_ij (rho = 0.78)
    # ------------------------------------------------------------------
    Q_ij = build_prior_weights(
        margX, margY, grid, copula_type="gaussian", rho=0.78
    )
    print(f"  Prior: Gaussian copula, rho=0.78, sum(Q_ij)={Q_ij.sum():.6f}")

    # ------------------------------------------------------------------
    # 5. Create DualModel (epsilon=1, beta=0, no exotic payoff)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. Solve dual optimisation
    # ------------------------------------------------------------------
    print("  Solving dual optimisation …")
    result = model.solve(tol=1e-6, max_iter=500, verbose=False)
    W_ij = result["W_ij"]
    print(f"  Converged: {result['converged']},  "
          f"iterations: {result['iterations']},  "
          f"|g|_inf = {result['final_grad_norm']:.3e}")

    # ------------------------------------------------------------------
    # 7. Compute calibration diagnostics
    # ------------------------------------------------------------------
    diag = compute_calibration_diagnostics(
        W_ij, grid["x_nodes"], grid["y_nodes"], market_data, epsilon=1.0
    )
    print(f"  Normalisation error   : {diag['normalization_error']:.3e}")
    print(f"  Max price error       : {diag['max_abs_price_error']:.3e}")
    print(f"  Max vol error (bps)   : {diag['max_abs_vol_error_bps']:.2f}")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    if save_results:
        os.makedirs("results", exist_ok=True)

        fig_smile = plot_smile_comparison(
            market_data, W_ij, grid["x_nodes"], grid["y_nodes"],
            title_suffix="Exp1 (ε=1, β=0)",
            save_path="results/exp1_smile.png",
        )
        fig_joint = plot_joint_distribution(
            W_ij, grid["x_nodes"], grid["y_nodes"],
            save_path="results/exp1_joint.png",
        )
        print("  Saved: results/exp1_smile.png, results/exp1_joint.png")

    # ------------------------------------------------------------------
    # 9. Save metrics JSON
    # ------------------------------------------------------------------
    results = {
        "scenario": "2s20s",
        "epsilon": 1.0,
        "beta": 0.0,
        "converged": bool(result["converged"]),
        "iterations": int(result["iterations"]),
        "final_grad_norm": float(result["final_grad_norm"]),
        "normalization_error": float(diag["normalization_error"]),
        "X_moment_error": float(diag["X_moment_error"]),
        "Y_moment_error": float(diag["Y_moment_error"]),
        "max_abs_price_error": float(diag["max_abs_price_error"]),
        "max_abs_vol_error_bps": float(diag["max_abs_vol_error_bps"]),
    }
    if save_results:
        with open("results/exp1_metrics.json", "w") as fh:
            json.dump(results, fh, indent=2)
        print("  Saved: results/exp1_metrics.json")

    return results


if __name__ == "__main__":
    run_exp1()
