"""
Tests for src/dual_model.py – DualModel and adaptive_epsilon_schedule.

All tests use a small n=10 quadrature grid for speed.
The DualModel is tested for:
- Initialisation
- Gradient correctness (finite-difference check)
- Hessian negative semidefiniteness
- Objective monotonically increases after a Newton step
- Non-negativity and normalisation of recovered W_ij
- Monotone decreasing model call prices
- Epsilon schedule properties
"""
import numpy as np
import pytest

from src.market_data import SyntheticCMSMarket
from src.marginals import MarginalDistribution
from src.quadrature import build_quadrature_grid
from src.copulas import build_prior_weights
from src.dual_model import DualModel, adaptive_epsilon_schedule


# ---------------------------------------------------------------------------
# Shared fixture: small n=10 DualModel for fast testing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    """Build a DualModel with n=10 quadrature nodes for fast testing."""
    mkt = SyntheticCMSMarket(scenario="2s20s", expiry_years=5.0)
    data = mkt.get_market_data()

    mX = MarginalDistribution(
        strikes=data["K_X"], call_prices=data["C_X"],
        forward=data["F_X"], T=data["T"],
    )
    mY = MarginalDistribution(
        strikes=data["K_Y"], call_prices=data["C_Y"],
        forward=data["F_Y"], T=data["T"],
    )

    grid = build_quadrature_grid(mX, mY, n=10)
    Q_ij = build_prior_weights(mX, mY, grid, copula_type="gaussian", rho=0.78)

    model = DualModel(
        x_nodes=grid["x_nodes"],
        y_nodes=grid["y_nodes"],
        x_weights=grid["x_weights"],
        y_weights=grid["y_weights"],
        Q_ij=Q_ij,
        K_X=data["K_X"],
        K_Y=data["K_Y"],
        K_Z=data["K_Z"],
        C_X_prices=data["C_X"],
        C_Y_prices=data["C_Y"],
        C_Z_prices=data["C_Z"],
        E_X=data["F_X"],
        E_Y=data["F_Y"],
        epsilon=1.0,
        beta=0.0,
    )
    return model, grid, data


@pytest.fixture(scope="module")
def solved_model(small_model):
    """Return (model, result) after running solve with loose tolerance."""
    model, grid, data = small_model
    result = model.solve(tol=1e-4, max_iter=100)
    return model, result, grid, data


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestDualModelInit:

    def test_dual_model_init(self, small_model):
        """DualModel initialises without error."""
        model, _, _ = small_model
        assert model is not None
        assert model._n_params == 3 + 20 + 20 + 20  # 3 + nK_X + nK_Y + nK_Z

    def test_dual_model_stores_nodes(self, small_model):
        """x_nodes and y_nodes are stored with the correct lengths."""
        model, grid, _ = small_model
        assert len(model.x_nodes) == len(grid["x_nodes"])
        assert len(model.y_nodes) == len(grid["y_nodes"])

    def test_dual_model_epsilon_property(self, small_model):
        """Epsilon property reads and writes correctly."""
        model, _, _ = small_model
        assert model.epsilon == pytest.approx(1.0)
        model.epsilon = 0.5
        assert model.epsilon == pytest.approx(0.5)
        model.epsilon = 1.0  # restore

    def test_dual_model_invalid_epsilon_raises(self, small_model):
        """Setting epsilon <= 0 raises ValueError."""
        model, _, _ = small_model
        with pytest.raises(ValueError):
            model.epsilon = -0.1
        model.epsilon = 1.0  # restore


# ---------------------------------------------------------------------------
# Objective and gradient
# ---------------------------------------------------------------------------

class TestObjectiveAndGradient:

    def test_objective_at_zero(self, small_model):
        """Dual objective at theta=0 is 0 (since A_ij=0 → L=-ε*1+ε=0)."""
        model, _, _ = small_model
        theta0 = np.zeros(model._n_params)
        L, _ = model.objective_and_gradient(theta0)
        assert abs(L) < 1e-12, f"Expected L=0 at theta=0, got {L:.6e}"

    def test_gradient_finite_differences(self, small_model):
        """Analytical gradient matches finite-difference approximation."""
        model, _, _ = small_model
        theta = np.zeros(model._n_params)
        _, grad_analytic = model.objective_and_gradient(theta)

        eps_fd = 1e-6
        grad_fd = np.empty_like(theta)
        for k in range(len(theta)):
            theta_p = theta.copy(); theta_p[k] += eps_fd
            theta_m = theta.copy(); theta_m[k] -= eps_fd
            L_p, _ = model.objective_and_gradient(theta_p)
            L_m, _ = model.objective_and_gradient(theta_m)
            grad_fd[k] = (L_p - L_m) / (2.0 * eps_fd)

        max_err = np.max(np.abs(grad_analytic - grad_fd))
        assert max_err < 1e-5, (
            f"Gradient FD error too large: {max_err:.3e}"
        )

    def test_gradient_at_zero_is_small(self, small_model):
        """Gradient at theta=0 has finite, reasonable magnitude."""
        model, _, _ = small_model
        theta = np.zeros(model._n_params)
        _, grad = model.objective_and_gradient(theta)
        assert np.all(np.isfinite(grad))
        assert np.max(np.abs(grad)) < 10.0  # should be much smaller than constraints


# ---------------------------------------------------------------------------
# Hessian
# ---------------------------------------------------------------------------

class TestHessian:

    def test_hessian_negative_semidefinite(self, small_model):
        """All eigenvalues of the Hessian H are ≤ 0 (H is NSD, L is concave)."""
        model, _, _ = small_model
        theta = np.zeros(model._n_params)
        H = model.hessian(theta)
        eigenvalues = np.linalg.eigvalsh(H)
        max_eig = float(eigenvalues.max())
        assert max_eig <= 1e-8, (
            f"Maximum Hessian eigenvalue {max_eig:.3e} > 0 (should be ≤ 0)"
        )

    def test_hessian_symmetric(self, small_model):
        """Hessian matrix is symmetric."""
        model, _, _ = small_model
        theta = np.zeros(model._n_params)
        H = model.hessian(theta)
        assert np.allclose(H, H.T, atol=1e-14), "Hessian is not symmetric"

    def test_hessian_shape(self, small_model):
        """Hessian has shape (n_params, n_params)."""
        model, _, _ = small_model
        theta = np.zeros(model._n_params)
        H = model.hessian(theta)
        p = model._n_params
        assert H.shape == (p, p)


# ---------------------------------------------------------------------------
# Newton step and objective increase
# ---------------------------------------------------------------------------

class TestNewtonStep:

    def test_objective_increases(self, small_model):
        """After one Newton step with line search, objective increases."""
        model, _, _ = small_model
        theta0 = np.zeros(model._n_params)
        L0, grad0 = model.objective_and_gradient(theta0)

        # Newton direction
        delta = model.newton_step(theta0)
        assert np.all(np.isfinite(delta)), "Newton direction contains inf/NaN"

        # Armijo line search: find alpha such that L increases
        armijo_c = 0.01
        slope = float(grad0 @ delta)

        alpha = 1.0
        L1 = L0
        for _ in range(30):
            theta1 = theta0 + alpha * delta
            L1, _ = model.objective_and_gradient(theta1)
            if L1 > L0 + armijo_c * alpha * max(slope, 0.0):
                break
            alpha *= 0.5

        assert L1 > L0, (
            f"Objective did not increase: L0={L0:.6f}, L1={L1:.6f}"
        )


# ---------------------------------------------------------------------------
# Post-solve checks
# ---------------------------------------------------------------------------

class TestSolveOutput:

    def test_w_ij_nonnegative(self, solved_model):
        """Recovered W_ij are all >= 0 after convergence."""
        _, result, _, _ = solved_model
        W = result["W_ij"]
        assert np.all(W >= -1e-14), (
            f"Negative W_ij: min = {W.min():.3e}"
        )

    def test_w_ij_normalized(self, solved_model):
        """sum(W_ij) ≈ 1 after convergence (within 1e-3)."""
        _, result, _, _ = solved_model
        W = result["W_ij"]
        total = W.sum()
        assert abs(total - 1.0) < 1e-3, (
            f"W_ij sums to {total:.6f}, expected ≈ 1"
        )

    def test_solve_returns_dict_keys(self, solved_model):
        """solve() returns a dict with all expected keys."""
        _, result, _, _ = solved_model
        for key in ("theta", "converged", "iterations",
                    "final_grad_norm", "W_ij", "L"):
            assert key in result, f"Missing key '{key}' in solve() output"

    def test_model_call_prices(self, solved_model):
        """Model call prices are non-increasing in strike."""
        model, result, _, data = solved_model
        W = result["W_ij"]
        prices_X = model.model_call_prices(W, data["K_X"], asset="X")
        diffs = np.diff(prices_X)
        assert np.all(diffs <= 1e-10), (
            f"Model X prices not non-increasing; max increase = {diffs.max():.3e}"
        )

    def test_model_call_prices_y(self, solved_model):
        """Model Y call prices are non-increasing in strike."""
        model, result, _, data = solved_model
        W = result["W_ij"]
        prices_Y = model.model_call_prices(W, data["K_Y"], asset="Y")
        diffs = np.diff(prices_Y)
        assert np.all(diffs <= 1e-10), (
            f"Model Y prices not non-increasing; max increase = {diffs.max():.3e}"
        )

    def test_model_call_prices_z(self, solved_model):
        """Model Z spread call prices are non-increasing in strike."""
        model, result, _, data = solved_model
        W = result["W_ij"]
        prices_Z = model.model_call_prices(W, data["K_Z"], asset="Z")
        diffs = np.diff(prices_Z)
        assert np.all(diffs <= 1e-10), (
            f"Model Z prices not non-increasing; max increase = {diffs.max():.3e}"
        )

    def test_compute_exotic_price_nonnegative(self, solved_model):
        """compute_exotic_price returns a non-negative price."""
        model, result, grid, data = solved_model
        W = result["W_ij"]
        x = grid["x_nodes"]
        y = grid["y_nodes"]
        K = data["F_X"] + data["F_Y"]
        payoff = np.maximum(x[:, np.newaxis] + y[np.newaxis, :] - K, 0.0)
        price = model.compute_exotic_price(W, payoff)
        assert price >= 0.0, f"Exotic price negative: {price}"


# ---------------------------------------------------------------------------
# Adaptive epsilon schedule
# ---------------------------------------------------------------------------

class TestAdaptiveEpsilonSchedule:

    def test_adaptive_epsilon_schedule(self):
        """Returns decreasing schedule from start to target."""
        schedule = adaptive_epsilon_schedule(
            epsilon_target=0.001, epsilon_start=1.0
        )
        assert len(schedule) >= 2
        assert schedule[0] == pytest.approx(1.0, rel=1e-6)
        assert schedule[-1] == pytest.approx(0.001, rel=1e-6)
        # Strictly decreasing
        assert np.all(np.diff(schedule) < 0.0), (
            "Schedule is not strictly decreasing"
        )

    def test_adaptive_epsilon_schedule_auto_start(self):
        """Without explicit start, auto-computes epsilon_start."""
        schedule = adaptive_epsilon_schedule(epsilon_target=0.01)
        assert len(schedule) >= 2
        assert schedule[-1] == pytest.approx(0.01, rel=1e-6)
        assert schedule[0] > schedule[-1]

    def test_adaptive_epsilon_schedule_raises_if_start_le_target(self):
        """Raises ValueError if epsilon_start <= epsilon_target."""
        with pytest.raises(ValueError):
            adaptive_epsilon_schedule(
                epsilon_target=1.0, epsilon_start=0.5
            )

    def test_adaptive_epsilon_schedule_raises_if_target_nonpositive(self):
        """Raises ValueError if epsilon_target <= 0."""
        with pytest.raises(ValueError):
            adaptive_epsilon_schedule(epsilon_target=0.0, epsilon_start=1.0)

    def test_adaptive_epsilon_schedule_values_bounded(self):
        """All schedule values lie in (epsilon_target, epsilon_start]."""
        eps_t, eps_s = 0.001, 1.0
        schedule = adaptive_epsilon_schedule(eps_t, eps_s)
        assert np.all(schedule >= eps_t - 1e-10)
        assert np.all(schedule <= eps_s + 1e-10)
