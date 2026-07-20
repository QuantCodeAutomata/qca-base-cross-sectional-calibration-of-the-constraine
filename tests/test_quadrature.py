"""
Tests for src/quadrature.py.

Verifies:
- Gauss-Legendre node count and weight properties
- Exact integration of polynomials
- build_quadrature_grid structure and shapes
"""
import numpy as np
import pytest

from src.market_data import SyntheticCMSMarket
from src.marginals import MarginalDistribution
from src.quadrature import (
    gauss_legendre_nodes_weights,
    map_to_interval,
    build_quadrature_grid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def marginals_20():
    """Build marginals using the 2s20s market with n=20 nodes."""
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
    return mX, mY, data


# ---------------------------------------------------------------------------
# Gauss-Legendre primitives
# ---------------------------------------------------------------------------

class TestGaussLegendreNodesWeights:

    def test_quadrature_node_count(self):
        """gauss_legendre_nodes_weights(n) returns exactly n nodes."""
        for n in [5, 10, 20, 50]:
            xi, w = gauss_legendre_nodes_weights(n)
            assert len(xi) == n, f"Expected {n} nodes, got {len(xi)}"
            assert len(w) == n

    def test_nodes_in_minus_one_to_one(self):
        """All nodes lie in (−1, 1)."""
        xi, _ = gauss_legendre_nodes_weights(10)
        assert np.all(xi > -1.0) and np.all(xi < 1.0)

    def test_weights_positive(self):
        """All GL weights are positive."""
        _, w = gauss_legendre_nodes_weights(10)
        assert np.all(w > 0.0)

    def test_weights_sum_to_two(self):
        """Weights on [−1, 1] sum to 2."""
        _, w = gauss_legendre_nodes_weights(10)
        assert abs(w.sum() - 2.0) < 1e-14

    def test_nodes_symmetric(self):
        """Nodes are symmetric around 0 for even n."""
        xi, _ = gauss_legendre_nodes_weights(10)
        xi_sorted = np.sort(xi)
        assert np.allclose(xi_sorted + xi_sorted[::-1], 0.0, atol=1e-14)


class TestMapToInterval:

    def test_quadrature_weights_sum(self):
        """Mapped weights sum to the interval length b − a."""
        for a, b in [(0.0, 1.0), (-2.5, 3.5), (0.1, 0.9)]:
            xi, w = gauss_legendre_nodes_weights(10)
            x_mapped, w_mapped = map_to_interval(xi, w, a, b)
            assert abs(w_mapped.sum() - (b - a)) < 1e-13, (
                f"Weight sum {w_mapped.sum():.6g} ≠ b-a={b-a}"
            )

    def test_mapped_nodes_in_interval(self):
        """Mapped nodes lie within [a, b]."""
        a, b = 0.01, 0.08
        xi, w = gauss_legendre_nodes_weights(15)
        x_mapped, _ = map_to_interval(xi, w, a, b)
        assert np.all(x_mapped >= a - 1e-14)
        assert np.all(x_mapped <= b + 1e-14)

    def test_quadrature_integrates_constant(self):
        """∫_a^b 1 dx = b − a (exact for any n)."""
        a, b = 0.01, 0.08
        xi, w = gauss_legendre_nodes_weights(5)
        x_mapped, w_mapped = map_to_interval(xi, w, a, b)
        result = np.sum(w_mapped * np.ones_like(x_mapped))
        assert abs(result - (b - a)) < 1e-13

    def test_quadrature_integrates_linear(self):
        """∫_0^1 x dx = 0.5 (exact for n ≥ 1)."""
        a, b = 0.0, 1.0
        xi, w = gauss_legendre_nodes_weights(3)
        x_mapped, w_mapped = map_to_interval(xi, w, a, b)
        result = np.dot(w_mapped, x_mapped)
        assert abs(result - 0.5) < 1e-14, f"Got {result:.15f}, expected 0.5"

    def test_quadrature_integrates_quadratic(self):
        """∫_0^1 x² dx = 1/3 (exact for n ≥ 2)."""
        a, b = 0.0, 1.0
        xi, w = gauss_legendre_nodes_weights(5)
        x_mapped, w_mapped = map_to_interval(xi, w, a, b)
        result = np.dot(w_mapped, x_mapped ** 2)
        assert abs(result - 1.0 / 3.0) < 1e-13


# ---------------------------------------------------------------------------
# build_quadrature_grid
# ---------------------------------------------------------------------------

class TestBuildQuadratureGrid:

    def test_build_grid_structure(self, marginals_20):
        """build_quadrature_grid returns dict with all expected keys."""
        mX, mY, _ = marginals_20
        grid = build_quadrature_grid(mX, mY, n=20)
        expected_keys = {
            "x_nodes", "y_nodes",
            "x_weights", "y_weights",
            "x_low", "x_high",
            "y_low", "y_high",
        }
        assert expected_keys.issubset(grid.keys()), (
            f"Missing keys: {expected_keys - set(grid.keys())}"
        )

    def test_build_grid_node_shapes(self, marginals_20):
        """x_nodes and y_nodes have shape (n,)."""
        mX, mY, _ = marginals_20
        n = 20
        grid = build_quadrature_grid(mX, mY, n=n)
        assert grid["x_nodes"].shape == (n,)
        assert grid["y_nodes"].shape == (n,)
        assert grid["x_weights"].shape == (n,)
        assert grid["y_weights"].shape == (n,)

    def test_build_grid_nodes_in_support(self, marginals_20):
        """All nodes lie within the declared support bounds."""
        mX, mY, _ = marginals_20
        grid = build_quadrature_grid(mX, mY, n=20)
        assert np.all(grid["x_nodes"] >= grid["x_low"] - 1e-12)
        assert np.all(grid["x_nodes"] <= grid["x_high"] + 1e-12)
        assert np.all(grid["y_nodes"] >= grid["y_low"] - 1e-12)
        assert np.all(grid["y_nodes"] <= grid["y_high"] + 1e-12)

    def test_build_grid_weights_positive(self, marginals_20):
        """Quadrature weights are all positive."""
        mX, mY, _ = marginals_20
        grid = build_quadrature_grid(mX, mY, n=20)
        assert np.all(grid["x_weights"] > 0.0)
        assert np.all(grid["y_weights"] > 0.0)

    def test_build_grid_weights_sum(self, marginals_20):
        """Weights sum equals the interval length x_high − x_low."""
        mX, mY, _ = marginals_20
        grid = build_quadrature_grid(mX, mY, n=20)
        width_x = grid["x_high"] - grid["x_low"]
        width_y = grid["y_high"] - grid["y_low"]
        assert abs(grid["x_weights"].sum() - width_x) < 1e-12
        assert abs(grid["y_weights"].sum() - width_y) < 1e-12

    def test_build_grid_integrates_constant(self, marginals_20):
        """Weighted sum of 1 ≈ x_high − x_low (measure of support)."""
        mX, mY, _ = marginals_20
        grid = build_quadrature_grid(mX, mY, n=20)
        result = np.sum(grid["x_weights"])
        width = grid["x_high"] - grid["x_low"]
        assert abs(result - width) < 1e-12
