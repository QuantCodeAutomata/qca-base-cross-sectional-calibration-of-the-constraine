"""
Experiment 2 – Small-epsilon payoff-tilted calibration.

Setup
-----
* Market  : 2s20s, T = 5Y, synthetic data
* Grid    : n = 200 Gauss-Legendre nodes
* Prior   : Gaussian copula, rho = 0.78
* Exotic  : call on X + Y at ATM strike K = E[X] + E[Y]
* Schedule: adaptive geometric annealing  ε ∈ {1.0 → 0.001}
* Model   : beta = -1 (lower-bound tilt), warm-start at each step
* Output  : calibration diagnostics, exotic price, saved metrics
"""
from __future__ import annotations

import json
import os

import numpy as np

from data.market_data_loader import load_cms_market_data
from src.marginals import MarginalDistribution
from src.quadrature import build_quadrature_grid
from src.copulas import build_prior_weights
from src.dual_model import DualModel, adaptive_epsilon_schedule
from src.reporting import compute_calibration_diagnostics


def run_exp2(save_results: bool = True) -> dict:
    """Run small-epsilon payoff-tilted calibration experiment.

    Parameters
    ----------
    save_results : bool
        Save metrics JSON to ``results/`` when *True*.

    Returns
    -------
    dict
    """
    print("=" * 60)
    print("Experiment 2: Small-ε payoff-tilted calibration  (β=−1)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load market data
    # ------------------------------------------------------------------
    market_data = load_cms_market_data(scenario="2s20s", expiry_years=5.0)

    # ------------------------------------------------------------------
    # 2. Marginal distributions
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
    # 3. Quadrature grid and prior
    # ------------------------------------------------------------------
    grid = build_quadrature_grid(margX, margY, n=200)
    Q_ij = build_prior_weights(
        margX, margY, grid, copula_type="gaussian", rho=0.78
    )

    # ------------------------------------------------------------------
    # 4. Exotic payoff  C_ij = max(x_i + y_j − K*, 0)
    #    where K* = E[X] + E[Y] = F_X + F_Y  (ATM for the sum)
    # ------------------------------------------------------------------
    K_exotic = market_data["F_X"] + market_data["F_Y"]
    x = grid["x_nodes"]
    y = grid["y_nodes"]
    exotic_payoff = np.maximum(
        x[:, np.newaxis] + y[np.newaxis, :] - K_exotic, 0.0
    )
    print(f"  Exotic K* = F_X + F_Y = {K_exotic:.4f}")

    # ------------------------------------------------------------------
    # 5. Adaptive epsilon schedule  1.0 → 0.001
    # ------------------------------------------------------------------
    schedule = adaptive_epsilon_schedule(
        epsilon_target=0.001, epsilon_start=1.0
    )
    print(f"  ε schedule: {schedule[0]:.4f} → {schedule[-1]:.6f} "
          f"in {len(schedule)} steps")

    # ------------------------------------------------------------------
    # 6. Warm-started annealing loop
    # ------------------------------------------------------------------
    theta_warm = None
    result = None

    for step, eps in enumerate(schedule):
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
            beta=-1.0,
            exotic_payoff_matrix=exotic_payoff,
        )
        result = model.solve(theta_init=theta_warm, tol=1e-6, max_iter=150)
        theta_warm = result["theta"]
        print(f"  ε={eps:.5f}  converged={result['converged']}  "
              f"|g|={result['final_grad_norm']:.2e}")

    W_ij = result["W_ij"]

    # ------------------------------------------------------------------
    # 7. Exotic price under calibrated model
    # ------------------------------------------------------------------
    exotic_price = model.compute_exotic_price(W_ij, exotic_payoff)
    print(f"  Exotic call price (K={K_exotic:.4f}): {exotic_price:.6f}")

    # ------------------------------------------------------------------
    # 8. Calibration diagnostics
    # ------------------------------------------------------------------
    diag = compute_calibration_diagnostics(
        W_ij, grid["x_nodes"], grid["y_nodes"], market_data,
        epsilon=float(schedule[-1]),
    )
    print(f"  Normalisation error   : {diag['normalization_error']:.3e}")
    print(f"  Max price error       : {diag['max_abs_price_error']:.3e}")
    print(f"  Max vol error (bps)   : {diag['max_abs_vol_error_bps']:.2f}")

    # ------------------------------------------------------------------
    # 9. Save metrics
    # ------------------------------------------------------------------
    results = {
        "scenario": "2s20s",
        "epsilon_final": float(schedule[-1]),
        "beta": -1.0,
        "K_exotic": float(K_exotic),
        "exotic_price": float(exotic_price),
        "converged": bool(result["converged"]),
        "iterations": int(result["iterations"]),
        "final_grad_norm": float(result["final_grad_norm"]),
        "normalization_error": float(diag["normalization_error"]),
        "max_abs_price_error": float(diag["max_abs_price_error"]),
        "max_abs_vol_error_bps": float(diag["max_abs_vol_error_bps"]),
        "epsilon_schedule": schedule.tolist(),
    }
    if save_results:
        os.makedirs("results", exist_ok=True)
        with open("results/exp2_metrics.json", "w") as fh:
            json.dump(results, fh, indent=2)
        print("  Saved: results/exp2_metrics.json")

    return results


if __name__ == "__main__":
    run_exp2()
