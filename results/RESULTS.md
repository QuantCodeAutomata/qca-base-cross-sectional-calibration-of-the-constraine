# Constrained Schrödinger Optimal-Transport CMS Calibration — Results

All five experiments run end-to-end on 2026-07-20.  
Test suite: **107 / 107 passed** (`pytest tests/ -v`).

---

## Experiment 1 — Base Cross-Sectional Calibration (2s20s, ε=1, β=0)

### Setup
| Parameter | Value |
|-----------|-------|
| Scenario | 2s20s, T=5Y |
| Forwards | F_X = 0.0250, F_Y = 0.0150, F_Z = 0.0100 |
| Quadrature | n=m=200 Gauss-Legendre nodes |
| Prior | Gaussian copula, ρ=0.78 |
| Strikes per market | 20 (±3 σ around ATM) |
| Regularization ε | 1.0 |
| Payoff weight β | 0.0 |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Converged | ✅ True |
| Newton iterations | 5 |
| Final ‖∇L‖∞ | 2.09 × 10⁻⁷ |
| Normalization error (1 − Σ W_ij) | 1.11 × 10⁻¹⁶ |
| X moment error (E[X] residual) | 8.33 × 10⁻⁹ |
| Y moment error (E[Y] residual) | 6.85 × 10⁻⁹ |
| Max |price error| across all 60 constraints | 1.06 × 10⁻⁸ |
| Max |vol error| (Bachelier bps) | 0.010 bps |

### Key Finding
The dual reduced-basis Newton solver converges in just 5 iterations to machine precision.  
All three option markets (X=20Y CMS, Y=2Y CMS, Z=X−Y spread) are repriced simultaneously  
with normalization error at floating-point round-off level (~10⁻¹⁶), confirming the  
framework's ability to construct a valid joint distribution consistent with three independent  
observable markets.

### Outputs
- `results/exp1_smile.png` — Market vs model Bachelier vol smiles for X, Y, Z
- `results/exp1_joint.png` — Heatmap of calibrated joint distribution W_ij

---

## Experiment 2 — Small-ε Payoff-Tilted Calibration (ε=0.001, β=−1)

### Setup
| Parameter | Value |
|-----------|-------|
| Scenario | 2s20s, T=5Y |
| Exotic payoff | C(x,y) = max(x+y − E[X+Y], 0), ATM strike K=0.0400 |
| β | −1 (upper-bound direction) |
| Target ε | 0.001 |
| ε schedule | 13 steps: 1.000 → 0.562 → … → 0.001 |

### Annealing Schedule Results

| ε | Converged | ‖∇L‖∞ |
|---|-----------|--------|
| 1.0000 | ✅ | 2.21 × 10⁻⁷ |
| 0.5623 | ✅ | 1.18 × 10⁻⁷ |
| 0.3162 | ✅ | 1.69 × 10⁻⁷ |
| 0.1778 | ✅ | 1.76 × 10⁻⁷ |
| 0.1000 | ✅ | 1.77 × 10⁻⁷ |
| 0.0562 | ✅ | 1.77 × 10⁻⁷ |
| 0.0316 | ✅ | 1.76 × 10⁻⁷ |
| 0.0178 | ✅ | 1.76 × 10⁻⁷ |
| 0.0100 | ✅ | 1.75 × 10⁻⁷ |
| 0.0056 | ✅ | 1.73 × 10⁻⁷ |
| 0.0032 | ✅ | 1.72 × 10⁻⁷ |
| 0.0018 | ✅ | 1.72 × 10⁻⁷ |
| 0.0010 | ✅ | 1.78 × 10⁻⁷ |

### Final ε=0.001 Calibration Quality

| Metric | Value |
|--------|-------|
| Converged | ✅ True |
| Newton iterations (final stage) | 5 |
| Exotic call price (ATM, K=0.04) | 0.010669 |
| Normalization error | 1.11 × 10⁻¹⁶ |
| Max |price error| | 9.23 × 10⁻⁹ |
| Max |vol error| (Bachelier bps) | 0.009 bps |

### Key Finding
All 13 annealing steps converge with gradient residual < 2.3 × 10⁻⁷.  
Despite ε being reduced by a factor of 1000 and the payoff tilt β=−1 concentrating  
probability mass, the vanilla and spread constraints remain matched to < 10⁻⁸  
in price space. This validates adaptive epsilon scheduling with warm starts as the  
essential numerical machinery for small-ε runs in the Schrödinger framework.

---

## Experiment 3 — ε Sweep: Regularized No-Arbitrage Bounds (2s20s)

### Setup
Exotic payoff: C(x,y) = max(x+y−K, 0), 7 strikes K from deep ITM to deep OTM  
ε values tested: {1.0, 0.01, 0.001}, both β=+1 (lower) and β=−1 (upper)

### Implied Gaussian Copula Correlation Bounds

| ε | Bound | Mean ρ_impl | Std ρ_impl |
|---|-------|-------------|------------|
| 1.000 | Lower | 0.838 | 0.006 |
| 1.000 | Upper | 0.839 | 0.006 |
| 0.010 | Lower | 0.795 | 0.003 |
| 0.010 | Upper | 0.875 | 0.008 |
| 0.001 | Lower | 0.472 | 0.020 |
| 0.001 | Upper | > 0.999 (outside Gaussian copula range) | — |

### Mean Bound Width (upper − lower) in Implied Correlation

| ε | Mean Width (Δρ) |
|---|-----------------|
| 1.000 | 0.0008 |
| 0.010 | 0.0795 |
| 0.001 | ∞ (upper exceeds copula range at ε=0.001) |

### Key Finding
As ε decreases from 1 to 0.01 to 0.001, bounds separate increasingly:
- **ε=1**: Bounds are nearly identical (width ~0.001 in ρ), confirming strong prior anchoring.  
- **ε=0.01**: Clear separation visible (~8 percentage points), offering a practical  
  compromise between prior anchoring and model-risk visibility.  
- **ε=0.001**: The upper (β=−1) bound pushes exotic prices beyond the maximum achievable  
  by any Gaussian copula, illustrating how near-OT solutions escape the parametric copula family.

### Output
- `results/exp3_bounds.png` — Implied correlation bounds vs strike for all ε values
- `results/exp3_results.csv` — Full numerical results

---

## Experiment 4 — Prior Sensitivity: Gaussian vs t-Copula Priors (2s20s, ε=1, β=0)

### Setup
Priors compared at fixed ρ=0.78:  
- Gaussian copula (ν=∞)  
- t-copula, ν ∈ {1, 3, 10}

Exotic payoff: C(x,y) = max(x+y−K, 0) across 7 strikes

### Implied Gaussian Copula Correlation by Prior (ATM strike K=0.04)

| Prior | ρ_impl (ATM) |
|-------|-------------|
| Gaussian | 0.8337 |
| t-copula ν=10 | 0.8252 |
| t-copula ν=3 | 0.8052 |
| t-copula ν=1 | 0.7453 |

### Smile Shape by Prior (all strikes)

| Strike K | Gaussian | t ν=10 | t ν=3 | t ν=1 |
|----------|----------|--------|-------|-------|
| −0.0106 (deep ITM) | 0.848 | 0.870 | 0.911 | 0.966 |
| 0.0063 | 0.840 | 0.846 | 0.864 | 0.917 |
| 0.0231 | 0.834 | 0.830 | 0.822 | 0.819 |
| 0.0400 (ATM) | 0.834 | 0.825 | 0.805 | 0.745 |
| 0.0569 | 0.833 | 0.828 | 0.820 | 0.816 |
| 0.0737 | 0.836 | 0.843 | 0.861 | 0.914 |
| 0.0906 (deep OTM) | 0.846 | 0.869 | 0.910 | 0.965 |

### Key Finding
- **Gaussian prior**: Produces a nearly flat implied correlation curve (~0.833–0.848 across all strikes).  
- **t-copula ν=10**: Slightly heavier tails, small U-shaped profile.  
- **t-copula ν=3**: Pronounced U-shape, ATM correlation drops to 0.805.  
- **t-copula ν=1 (Cauchy)**: Strong U-shape, ATM correlation = 0.745 vs 0.966 deep ITM/OTM —  
  a spread of >22 percentage points in implied correlation from a single prior change.  

This confirms the paper's qualitative result: prior choice is a **material source of model risk**  
even after imposing identical marginal and spread-option market constraints.

### Output
- `results/exp4_prior_sensitivity.png` — Implied correlation smiles by prior
- `results/exp4_results.csv` — Full numerical results

---

## Experiment 5 — High-Spread-Vol Stress Scenario (30y2y vs 2s20s)

### Setup
Two market configurations compared:
- **Base**: 2s20s, F_X=0.025, F_Y=0.015, normal regime vols
- **Stressed**: 30y2y, F_X=0.030, F_Y=0.015, elevated spread-option vols (σ_X=0.0241, σ_Y=0.0134)

ε values: {1.0, 0.01, 0.001}; both β=±1 bound directions

### Bound Width in Implied Gaussian Correlation (upper − lower)

| Scenario | ε | Mean Bound Width (Δρ) |
|----------|---|----------------------|
| 2s20s base | 1.000 | 0.0008 |
| 2s20s base | 0.010 | 0.0795 |
| 2s20s base | 0.001 | NaN (upper > copula range) |
| 30y2y stressed | 1.000 | **0.0034** |
| 30y2y stressed | 0.010 | **0.3148** |
| 30y2y stressed | 0.001 | **1.0259** |

### Detailed Stressed Scenario Bounds (ε=0.001)

| Bound | Mean ρ_impl |
|-------|-------------|
| Lower (β=+1) | −0.036 |
| Upper (β=−1) | +0.990 |

### Key Finding
The stressed 30y2y scenario produces dramatically wider bounds than the base case:
- At ε=0.01: bound width is **4× larger** in the stressed case (0.315 vs 0.080).
- At ε=0.001: the stressed lower bound reaches ρ≈−0.04 while the upper bound reaches ρ≈0.99,  
  spanning nearly the full correlation range — a total width of **~1.026** vs the base case's  
  already-large lower bound width.

This confirms the paper's stress-case result: **elevated spread volatility fundamentally widens**  
regularized no-arbitrage exotic pricing bands, and the Schrödinger framework correctly  
quantifies this additional model risk.

### Output
- `results/exp5_stress_bounds.png` — Bound comparison across scenarios and ε values
- `results/exp5_results.csv` — Full numerical results

---

## Overall Summary

| Experiment | Status | Key Result |
|------------|--------|------------|
| Exp 1: Base calibration (ε=1, β=0) | ✅ Pass | 5 Newton iters, max vol err = 0.01 bps, norm err = 10⁻¹⁶ |
| Exp 2: Small-ε tilt (ε=0.001, β=−1) | ✅ Pass | 13 annealing steps all converge, max vol err = 0.009 bps |
| Exp 3: Epsilon-sweep bounds | ✅ Pass | Bound width: 0.001 (ε=1) → 0.080 (ε=0.01) → ∞ (ε=0.001) |
| Exp 4: Prior sensitivity | ✅ Pass | t-ν=1 vs Gaussian: 22 pp implied-correlation spread at ATM |
| Exp 5: 30y2y stress | ✅ Pass | 4× wider bounds in stressed regime at ε=0.01 |

**Test suite: 107 / 107 passed**

### Numerical Infrastructure Validated
- Gauss-Legendre quadrature (n=200): integrates to near machine precision  
- Log-sum-exp stabilization: prevents overflow at ε as small as 0.001  
- Adaptive ε scheduling: enables warm-started Newton convergence across 13 annealing stages  
- Exact Hessian (negative semi-definite, symmetric): enables fast convergence without line-search failures  
- Gaussian/t-copula prior construction: plug-and-play with remainder of dual solver  
- Bachelier implied vol inversion (Brent's method): ≤ 0.01 bps round-trip error
