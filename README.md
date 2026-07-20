# Constrained Schrödinger Optimal-Transport CMS Calibration

A Python implementation of the **dual constrained Schrödinger optimal-transport
framework** for joint calibration of bivariate CMS rate distributions to three
simultaneous markets: CMS options on X (long-tenor rate), CMS options on Y
(short-tenor rate), and CMS spread options on Z = X − Y.

## Overview

The model constructs a joint probability distribution W_ij over (X, Y) that:
1. Exactly matches marginal option smiles for X and Y (all 20 strikes each).
2. Exactly matches the spread-option smile for Z = X − Y (20 strikes).
3. Stays close to a copula-based prior Q_ij via entropic regularisation with parameter ε.
4. Optionally tilts toward/away from an exotic payoff C(x,y) via weight β.

The calibration is solved as a **reduced-basis dual problem** using an exact
gradient and Hessian with a damped Newton method, optionally with an adaptive
ε-annealing schedule for small-ε runs.

## Repository Structure

```
├── data/                   # Market data loading utilities
│   └── market_data_loader.py
├── src/                    # Core implementation
│   ├── market_data.py      # Synthetic CMS market snapshot generation
│   ├── marginals.py        # Arbitrage-aware smile interpolation → CDFs/PDFs
│   ├── quadrature.py       # Gauss-Legendre quadrature on truncated supports
│   ├── copulas.py          # Gaussian & t-copula prior densities
│   ├── dual_model.py       # Dual objective, gradient, Hessian, Newton solver
│   └── reporting.py        # Bachelier vol conversion, smile plots, diagnostics
├── exp/                    # Experiment scripts
│   ├── exp1_base_calibration.py    # ε=1, β=0 baseline
│   ├── exp2_small_epsilon.py       # ε=0.001, β=−1, ATM option on X+Y
│   ├── exp3_epsilon_sweep.py       # Bound sweep over strikes and ε values
│   ├── exp4_prior_sensitivity.py   # Gaussian vs t-copula prior comparison
│   └── exp5_stress_scenario.py     # 30y2y stressed-spread-vol scenario
├── tests/                  # pytest test suite (107 tests)
├── results/                # Saved metrics, plots, and RESULTS.md
└── pyproject.toml
```

## Experiments

| # | Title | Key Config | Main Finding |
|---|-------|------------|--------------|
| 1 | Base calibration | 2s20s, ε=1, β=0 | 5 Newton iters; vol err < 0.01 bps |
| 2 | Small-ε payoff tilt | 2s20s, ε=0.001, β=−1 | 13 annealing stages; vol err < 0.01 bps |
| 3 | ε-sweep bounds | 2s20s, ε∈{1,0.01,0.001} | Bounds widen from 0.001→0.08→∞ in Δρ |
| 4 | Prior sensitivity | Gaussian vs t-ν∈{1,3,10} | t-ν=1: 22 pp implied-ρ swing at ATM |
| 5 | 30y2y stress | Stressed spread vol | 4× wider bounds vs base at ε=0.01 |

## Installation

```bash
pip install -e .
# or
pip install numpy scipy matplotlib seaborn pandas
```

## Usage

```bash
# Run all experiments
python exp/exp1_base_calibration.py
python exp/exp2_small_epsilon.py
python exp/exp3_epsilon_sweep.py
python exp/exp4_prior_sensitivity.py
python exp/exp5_stress_scenario.py

# Run full test suite
pytest tests/ -v
```

## Methodology

Based on the constrained Schrödinger bridge / entropic optimal-transport
framework applied to CMS rate bivariate pricing.

**Dual objective** (maximised over θ = {m₀, m₁, m₂, {qᵣˣ}, {qᵣʸ}, {qᵣᶻ}}):

```
L(θ) = m₀ + m₁·E[X] + m₂·E[Y]
      + Σᵣ qᵣˣ·Cˣ(r) + Σᵣ qᵣʸ·Cʸ(r) + Σᵣ qᵣᶻ·Cᶻ(r)
      − ε · Σᵢⱼ Qᵢⱼ · exp(Aᵢⱼ(θ)/ε)  +  ε
```

where `A_ij(θ) = m₀ + m₁xᵢ + m₂yⱼ + Σ qᵣˣ(xᵢ−r)⁺ + Σ qᵣʸ(yⱼ−r)⁺ + Σ qᵣᶻ(xᵢ−yⱼ−r)⁺ − β·Cᵢⱼ`.

The calibrated joint distribution is `W_ij = Q_ij · exp(A_ij(θ*)/ε)`.

**Copula priors:**

- *Gaussian*: `c(u,v;ρ)` — standard bivariate normal density evaluated at (Φ⁻¹(u), Φ⁻¹(v))
- *t-copula*: `c(u,v;ρ,ν)` with Student-t marginals and bivariate t density

**Solver:** Damped Newton with exact gradient and Hessian; adaptive ε-annealing
for ε < 0.05; log-sum-exp stabilisation throughout.

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full tables and plots.

Key metrics (Exp 1, baseline):
- Normalization error: **1.11 × 10⁻¹⁶** (machine precision)
- Max vol error: **0.010 bps**
- Newton iterations to convergence: **5**
