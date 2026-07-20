"""
Tests for src/copulas.py.

Verifies:
- gaussian_copula_density: positivity, independence case, range
- t_copula_density: positivity, large-ν Gaussian limit
- build_prior_weights: non-negativity, normalisation
"""
import numpy as np
import pytest

from src.market_data import SyntheticCMSMarket
from src.marginals import MarginalDistribution
from src.quadrature import build_quadrature_grid
from src.copulas import gaussian_copula_density, t_copula_density, build_prior_weights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_grid():
    """Return (margX, margY, grid) built with n=20 nodes."""
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
    grid = build_quadrature_grid(mX, mY, n=20)
    return mX, mY, grid


# ---------------------------------------------------------------------------
# Gaussian copula density
# ---------------------------------------------------------------------------

class TestGaussianCopulaDensity:

    def test_gaussian_copula_density_positive(self):
        """Gaussian copula density > 0 for valid (u, v, rho) inputs."""
        for u, v, rho in [(0.3, 0.7, 0.5), (0.1, 0.9, -0.3), (0.5, 0.5, 0.0)]:
            d = gaussian_copula_density(u, v, rho)
            assert d > 0.0, f"density={d} for u={u}, v={v}, rho={rho}"

    def test_gaussian_copula_density_independent(self):
        """rho = 0 gives density = 1 for all (u, v) in (0, 1)."""
        for u, v in [(0.3, 0.7), (0.5, 0.5), (0.1, 0.9), (0.8, 0.2)]:
            d = gaussian_copula_density(u, v, rho=0.0)
            assert abs(d - 1.0) < 1e-10, (
                f"Expected c=1 for rho=0, got {d:.8f} at u={u}, v={v}"
            )

    def test_gaussian_copula_density_symmetric(self):
        """c(u, v; rho) = c(v, u; rho)."""
        u, v, rho = 0.3, 0.7, 0.5
        assert abs(
            gaussian_copula_density(u, v, rho) - gaussian_copula_density(v, u, rho)
        ) < 1e-12

    def test_gaussian_copula_density_array_input(self):
        """Scalar and array inputs both work."""
        u = np.array([0.2, 0.5, 0.8])
        v = np.array([0.8, 0.5, 0.2])
        d = gaussian_copula_density(u, v, rho=0.4)
        assert d.shape == (3,)
        assert np.all(d > 0)

    def test_gaussian_copula_density_positive_rho_increases_density_near_diagonal(self):
        """Positive correlation increases density near the main diagonal."""
        u, v = 0.6, 0.6  # near diagonal
        d_pos = gaussian_copula_density(u, v, rho=0.8)
        d_ind = gaussian_copula_density(u, v, rho=0.0)
        assert d_pos > d_ind, "Positive rho should inflate diagonal density"

    def test_gaussian_copula_density_broadcast(self):
        """2D broadcasting: shape (n,1) × (1,m) → (n,m)."""
        u = np.linspace(0.1, 0.9, 5)[:, np.newaxis]  # (5, 1)
        v = np.linspace(0.1, 0.9, 4)[np.newaxis, :]  # (1, 4)
        d = gaussian_copula_density(u, v, rho=0.5)
        assert d.shape == (5, 4)
        assert np.all(d > 0)


# ---------------------------------------------------------------------------
# Student-t copula density
# ---------------------------------------------------------------------------

class TestTCopulaDensity:

    def test_t_copula_density_positive(self):
        """t copula density > 0 for valid inputs."""
        for nu in [1.0, 3.0, 10.0]:
            d = t_copula_density(0.3, 0.7, rho=0.5, nu=nu)
            assert d > 0.0, f"Non-positive t-copula density for nu={nu}"

    def test_t_copula_density_limits_gaussian(self):
        """Large ν approximates the Gaussian copula density within 1 %."""
        u, v, rho = 0.4, 0.6, 0.5
        d_gauss = gaussian_copula_density(u, v, rho)
        d_t_large = t_copula_density(u, v, rho, nu=1000.0)
        rel_err = abs(d_t_large - d_gauss) / d_gauss
        assert rel_err < 0.01, (
            f"t-copula(ν=1000) vs Gaussian: relative error {rel_err:.3e}"
        )

    def test_t_copula_density_nu1_heavier_tails(self):
        """t-copula (ν=1) has heavier tails: more mass at (0.05, 0.05)."""
        u, v, rho = 0.05, 0.05, 0.5  # lower-left tail
        d_gauss = gaussian_copula_density(u, v, rho)
        d_t1 = t_copula_density(u, v, rho, nu=1.0)
        assert d_t1 > d_gauss, "t(ν=1) should have more tail mass than Gaussian"

    def test_t_copula_density_symmetric(self):
        """c_t(u, v; rho, nu) = c_t(v, u; rho, nu)."""
        u, v, rho, nu = 0.3, 0.7, 0.5, 4.0
        assert abs(
            t_copula_density(u, v, rho, nu) - t_copula_density(v, u, rho, nu)
        ) < 1e-10

    def test_t_copula_density_independent(self):
        """rho = 0 gives t-copula density near 1 at the centre.

        Unlike the Gaussian copula, the t-copula does not factorise
        exactly at rho=0 (due to tail dependence), so the density at
        (0.5, 0.5) is close to – but not exactly – 1.  We allow up
        to 15 % deviation.
        """
        d = t_copula_density(0.5, 0.5, rho=0.0, nu=10.0)
        assert abs(d - 1.0) < 0.15


# ---------------------------------------------------------------------------
# build_prior_weights
# ---------------------------------------------------------------------------

class TestBuildPriorWeights:

    def test_prior_weights_nonnegative(self, small_grid):
        """Q_ij values are all >= 0."""
        mX, mY, grid = small_grid
        Q = build_prior_weights(mX, mY, grid, copula_type="gaussian", rho=0.5)
        assert np.all(Q >= 0.0), f"Negative Q_ij: min={Q.min():.3e}"

    def test_prior_weights_normalized(self, small_grid):
        """sum(Q_ij) ≈ 1 (within 5 %)."""
        mX, mY, grid = small_grid
        Q = build_prior_weights(mX, mY, grid, copula_type="gaussian", rho=0.5)
        total = Q.sum()
        assert abs(total - 1.0) < 0.05, (
            f"Q_ij sums to {total:.4f}, expected ≈ 1"
        )

    def test_prior_weights_shape(self, small_grid):
        """Q_ij has shape (n, n)."""
        mX, mY, grid = small_grid
        n = len(grid["x_nodes"])
        Q = build_prior_weights(mX, mY, grid, copula_type="gaussian", rho=0.78)
        assert Q.shape == (n, n)

    def test_prior_weights_gaussian_t_different(self, small_grid):
        """Gaussian and t-copula priors produce different Q_ij matrices."""
        mX, mY, grid = small_grid
        Q_gauss = build_prior_weights(
            mX, mY, grid, copula_type="gaussian", rho=0.5
        )
        Q_t = build_prior_weights(
            mX, mY, grid, copula_type="t", rho=0.5, nu=3.0
        )
        assert not np.allclose(Q_gauss, Q_t), (
            "Gaussian and t-copula should produce different prior weights"
        )

    def test_prior_weights_t_large_nu_near_gaussian(self, small_grid):
        """t-copula (ν=1000) prior ≈ Gaussian copula prior."""
        mX, mY, grid = small_grid
        Q_gauss = build_prior_weights(
            mX, mY, grid, copula_type="gaussian", rho=0.5
        )
        Q_t_large = build_prior_weights(
            mX, mY, grid, copula_type="t", rho=0.5, nu=1000.0
        )
        max_rel_err = np.max(np.abs(Q_gauss - Q_t_large)) / Q_gauss.max()
        assert max_rel_err < 0.01, (
            f"t(ν=1000) prior differs from Gaussian by {max_rel_err:.3e}"
        )

    def test_prior_weights_independent_copula_factorises(self, small_grid):
        """For rho=0 (Gaussian copula), Q_ij ≈ p_x_i * p_y_j."""
        mX, mY, grid = small_grid
        Q = build_prior_weights(mX, mY, grid, copula_type="gaussian", rho=0.0)
        # Row sums and column sums should be approximately equal (uniform-ish)
        row_sums = Q.sum(axis=1)
        col_sums = Q.sum(axis=0)
        # For independent copula: Q_ij = p_x_i * p_y_j → outer product
        assert np.allclose(Q, np.outer(row_sums, col_sums), rtol=0.1)

    def test_prior_weights_t_requires_nu(self, small_grid):
        """t-copula raises ValueError when nu is not provided."""
        mX, mY, grid = small_grid
        with pytest.raises(ValueError):
            build_prior_weights(mX, mY, grid, copula_type="t", rho=0.5)

    def test_prior_weights_unknown_copula_raises(self, small_grid):
        """Unknown copula_type raises ValueError."""
        mX, mY, grid = small_grid
        with pytest.raises(ValueError):
            build_prior_weights(mX, mY, grid, copula_type="frank", rho=0.5)
