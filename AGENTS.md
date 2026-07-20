# AGENTS.md – Repository Knowledge

## Project Overview
Constrained Schrödinger Optimal Transport model for CMS (Constant Maturity Swap)
rate calibration.  Implements the dual parameterisation and damped-Newton solver
for the entropic OT calibration problem.

## Module Structure (`src/`)

| File | Purpose |
|------|---------|
| `market_data.py` | Synthetic CMS option market data (Bachelier model, quadratic smile) |
| `marginals.py` | Breeden-Litzenberger marginal density extraction |
| `quadrature.py` | Gauss-Legendre quadrature mapping to arbitrary intervals |
| `copulas.py` | Gaussian / Student-t copula densities + discrete prior construction |
| `dual_model.py` | **Core**: dual OT model, damped-Newton solver, epsilon annealing |
| `reporting.py` | Diagnostics (price + vol errors), smile plots, implied-rho inversion |

## Key Design Decisions

### Bachelier (Normal) Model
- All options use Bachelier (normal) pricing — appropriate for rates that can go negative.
- `bachelier_call_price(F, K, sigma, T)` handles the ATM case `K=F` exactly as
  `sigma*sqrt(T)/sqrt(2*pi)` to avoid 0/0 in the formula.
- `bachelier_implied_vol` uses `scipy.optimize.brentq`; bracketing expands if needed.

### Marginal Distribution (`MarginalDistribution`)
- Input: market (K, C) pairs → invert to Bachelier implied vols.
- Regularise smile via degree-2 polynomial fit → `CubicSpline` on smoothed values.
- Fine grid: 3 000 points spanning `[K_min - 2*(K_max-K_min), K_max + 2*(K_max-K_min)]`
  (5× the original strike range).  Vol outside `[K_min, K_max]` is extrapolated flat
  (constant boundary value from the CubicSpline) via `cs(np.clip(K_fine, K_in[0], K_in[-1]))`.
- Breeden-Litzenberger with central finite differences on the extended fine grid.
- Density clipped to ≥0 and normalised; CDF built by `cumulative_trapezoid`.
- `_x_low` / `_x_high` (ppf search bounds) are pinned to the **original** `[K_in[0], K_in[-1]]`
  to keep ppf output within the market-data strike range.
- `_cdf_at_x_low` / `_cdf_at_x_high` precomputed; `_ppf_raw` clamps `p` to this range
  before calling `brentq`, so no bracket error for tail quantiles.
- `ppf` uses `brentq` on the CDF interpolant; edge cases guarded by `q_trunc`.
- `conftest.py` at repo root inserts the project directory into `sys.path`.

### Dual Model (`DualModel`)
- Dual variable: `theta = [m0, m1, m2, q_X[nK_X], q_Y[nK_Y], q_Z[nK_Z]]`
- Basis tensor `Phi` of shape `(n_params, n, m)` is precomputed for efficiency.
- **Numerical stability**: A_ij/ε is shifted by its maximum before exponentiation.
- Objective is **concave** (Hessian is negative semi-definite); we **maximise** L.
- Newton direction: `H @ delta = -g`, solved via `linalg.solve` with small diagonal regularisation.
- Armijo backtracking (c=0.01, max 30 halvings) guarantees ascent.
- `epsilon` is a settable property to support annealing from outside the class.
- `adaptive_epsilon_schedule` returns a geometric sequence from `eps_start` to `eps_target`.

### Copulas
- Gaussian copula density at `rho=0` is exactly 1 (verified in tests).
- t-copula uses `scipy.special.gammaln` for numerical stability of the log-density.
- `build_prior_weights` is fully vectorised (meshgrid broadcast, no Python loops).

### Reporting
- `prices_to_bachelier_vols` returns NaN on failure (handles deep OTM gracefully).
- `gaussian_copula_price` prices **max(X + Y − K, 0)** (sum call, not spread).
- `compute_implied_gaussian_correlation` inverts with `brentq`; raises `ValueError`
  if the target price is outside the range achievable by the Gaussian copula.

## Scenarios

| Scenario | F_X | F_Y | sigma_X_atm | sigma_Z_atm |
|----------|-----|-----|-------------|-------------|
| `2s20s`  | 2.5 % | 1.5 % | 70 bps | 40 bps |
| `30y2y`  | 3.0 % | 1.5 % | 90 bps | 80 bps |

`stressed=True` amplifies all vols and skews by 20 %.

## Typical Workflow
```python
from src.market_data import SyntheticCMSMarket
from src.marginals import MarginalDistribution
from src.quadrature import build_quadrature_grid
from src.copulas import build_prior_weights
from src.dual_model import DualModel, adaptive_epsilon_schedule
from src.reporting import compute_calibration_diagnostics

mkt  = SyntheticCMSMarket('2s20s', expiry_years=5.0)
data = mkt.get_market_data()
marg_X = MarginalDistribution(data['K_X'], data['C_X'], data['F_X'], data['T'])
marg_Y = MarginalDistribution(data['K_Y'], data['C_Y'], data['F_Y'], data['T'])
quad   = build_quadrature_grid(marg_X, marg_Y, n=200)
Q_ij   = build_prior_weights(marg_X, marg_Y, quad, copula_type='gaussian', rho=0.78)

schedule = adaptive_epsilon_schedule(epsilon_target=0.001)
dm = DualModel(..., epsilon=schedule[0])
theta = None
for eps in schedule:
    dm.epsilon = eps
    res = dm.solve(theta_init=theta, max_iter=100, tol=1e-8)
    theta = res['theta']
W_ij = res['W_ij']
```

## Observed Numerical Behaviour
- Moments (E[X], E[Y]) are recovered to ~1e-6 error after ~50 Newton iterations.
- Maximum call price error ~5e-6 (in rate units).
- Maximum vol error can be large in bps for deep OTM strikes (tiny prices) — this
  is expected and not a bug.
- Convergence is declared when `max|grad| < tol`; typical tol=1e-6 to 1e-8.

## Tests (`tests/`)

| File | What it tests | Notes |
|------|---------------|-------|
| `test_market_data.py` | Bachelier pricing, implied vol, `SyntheticCMSMarket` | 18 tests |
| `test_marginals.py` | CDF/PDF/PPF properties for `MarginalDistribution` | 12 tests |
| `test_quadrature.py` | GL nodes/weights, `map_to_interval`, `build_quadrature_grid` | 13 tests |
| `test_copulas.py` | Gaussian + t-copula densities, `build_prior_weights` | 19 tests |
| `test_dual_model.py` | `DualModel` init, FD gradient check, Hessian NSD, Newton step | 23 tests |
| `test_reporting.py` | Vol roundtrip, diagnostics, copula pricing, implied rho | 15 tests |

- Run all tests: `python -m pytest tests/ -v` (finishes in ~3 s, n=10–20 grids)
- t-copula density at (0.5, 0.5, rho=0, nu=10) ≈ 1.051 (not exactly 1 — t-copula is
  NOT the independence copula at rho=0 due to tail dependence).

## Experiments (`exp/`)

| Script | Setup | Key parameters |
|--------|-------|----------------|
| `exp1_base_calibration.py` | 2s20s, ε=1, β=0 | n=200 GL nodes |
| `exp2_small_epsilon.py` | 2s20s, annealing ε→0.001, β=−1, ATM exotic | warm start per step |
| `exp3_epsilon_sweep.py` | 2s20s, ε∈{1,0.01,0.001}, β=±1, 7 strikes | n=100 for speed |
| `exp4_prior_sensitivity.py` | 2s20s, 4 priors (Gaussian + t-copula ν=1,3,10) | n=100 |
| `exp5_stress_scenario.py` | 30y2y stressed + 2s20s base, ε sweep | n=100 |

Outputs go to `results/` (JSON, CSV, PNG). Pass `save_results=False` to skip disk I/O.

## data/market_data_loader.py
Thin wrapper around `SyntheticCMSMarket`.  Adds `'source'`, `'scenario'`, and `'note'`
keys to the returned dict.  CMS option data is not available from general-purpose
equity-focused APIs; synthetic generation is the documented production path.

## Dependencies
numpy, scipy, matplotlib, pandas (all standard; no uncommon packages).
