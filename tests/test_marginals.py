"""
Tests for src/marginals.py – MarginalDistribution.

Uses the 2s20s synthetic market data to build marginals and
verifies standard probability-distribution properties.
"""
import numpy as np
import pytest

from src.market_data import SyntheticCMSMarket
from src.marginals import MarginalDistribution


@pytest.fixture(scope="module")
def marginals():
    """Return (margX, margY, data) built from 2s20s synthetic market."""
    mkt = SyntheticCMSMarket(scenario="2s20s", expiry_years=5.0)
    data = mkt.get_market_data()
    mX = MarginalDistribution(
        strikes=data["K_X"],
        call_prices=data["C_X"],
        forward=data["F_X"],
        T=data["T"],
    )
    mY = MarginalDistribution(
        strikes=data["K_Y"],
        call_prices=data["C_Y"],
        forward=data["F_Y"],
        T=data["T"],
    )
    return mX, mY, data


# ---------------------------------------------------------------------------
# CDF properties
# ---------------------------------------------------------------------------

class TestCDF:

    def test_cdf_bounds(self, marginals):
        """CDF values lie in [0, 1] for any input x."""
        mX, _, data = marginals
        x_test = np.linspace(data["K_X"][0], data["K_X"][-1], 50)
        cdf_vals = mX.cdf(x_test)
        assert np.all(cdf_vals >= 0.0 - 1e-12)
        assert np.all(cdf_vals <= 1.0 + 1e-12)

    def test_cdf_monotone(self, marginals):
        """CDF is non-decreasing."""
        mX, _, data = marginals
        x_test = np.linspace(data["K_X"][0], data["K_X"][-1], 100)
        cdf_vals = mX.cdf(x_test)
        diffs = np.diff(cdf_vals)
        assert np.all(diffs >= -1e-10), (
            f"CDF not monotone; min diff = {diffs.min():.3e}"
        )

    def test_cdf_boundary_values(self, marginals):
        """CDF ≈ 0 at left boundary and ≈ 1 at right boundary."""
        mX, _, data = marginals
        K = data["K_X"]
        assert mX.cdf(K[0]) < 0.1   # small near left edge
        assert mX.cdf(K[-1]) > 0.9  # large near right edge

    def test_cdf_scalar_and_array_output(self, marginals):
        """Scalar input returns float; array input returns ndarray."""
        mX, _, data = marginals
        x_mid = 0.5 * (data["K_X"][0] + data["K_X"][-1])
        scalar_out = mX.cdf(x_mid)
        assert np.ndim(scalar_out) == 0, "Expected scalar output for scalar input"

        arr_out = mX.cdf(np.array([x_mid, x_mid + 0.001]))
        assert arr_out.ndim == 1


# ---------------------------------------------------------------------------
# PDF properties
# ---------------------------------------------------------------------------

class TestPDF:

    def test_pdf_nonnegative(self, marginals):
        """PDF values are >= 0 everywhere on the support."""
        mX, _, data = marginals
        x_test = np.linspace(data["K_X"][0], data["K_X"][-1], 200)
        pdf_vals = mX.pdf(x_test)
        assert np.all(pdf_vals >= -1e-12), (
            f"Negative PDF; min = {pdf_vals.min():.3e}"
        )

    def test_pdf_integrates_to_one(self, marginals):
        """Numerical integral of PDF over the support is within 5 % of 1."""
        mX, _, data = marginals
        K_fine = np.linspace(data["K_X"][0], data["K_X"][-1], 1000)
        pdf_vals = mX.pdf(K_fine)
        integral = np.trapz(pdf_vals, K_fine)
        assert abs(integral - 1.0) < 0.05, (
            f"PDF integral = {integral:.4f}, expected ≈ 1"
        )

    def test_pdf_zero_outside_support(self, marginals):
        """PDF returns 0 outside the strike range."""
        mX, _, data = marginals
        far_left = data["K_X"][0] - 1.0
        far_right = data["K_X"][-1] + 1.0
        assert mX.pdf(far_left) == pytest.approx(0.0, abs=1e-10)
        assert mX.pdf(far_right) == pytest.approx(0.0, abs=1e-10)

    def test_pdf_atm_positive(self, marginals):
        """PDF at the forward rate is strictly positive."""
        mX, _, data = marginals
        assert mX.pdf(data["F_X"]) > 0.0


# ---------------------------------------------------------------------------
# PPF (quantile function) properties
# ---------------------------------------------------------------------------

class TestPPF:

    def test_ppf_cdf_roundtrip(self, marginals):
        """ppf(cdf(x)) ≈ x for interior points (relative error < 1 %)."""
        mX, _, data = marginals
        K = data["K_X"]
        # Use interior 60 % of the support to avoid boundary effects
        x_lo = K[0] + 0.2 * (K[-1] - K[0])
        x_hi = K[-1] - 0.2 * (K[-1] - K[0])
        x_test = np.linspace(x_lo, x_hi, 10)

        for x in x_test:
            u = mX.cdf(x)
            x_back = mX.ppf(u)
            rel_err = abs(x - x_back) / max(abs(x), 1e-6)
            assert rel_err < 0.01, (
                f"ppf(cdf({x:.4f})) = {x_back:.4f}, relative error = {rel_err:.3e}"
            )

    def test_ppf_monotone(self, marginals):
        """ppf is non-decreasing on (0, 1)."""
        mX, _, _ = marginals
        u_test = np.linspace(0.05, 0.95, 50)
        ppf_vals = mX.ppf(u_test)
        diffs = np.diff(ppf_vals)
        assert np.all(diffs >= -1e-10), (
            f"PPF not monotone; min diff = {diffs.min():.3e}"
        )

    def test_ppf_output_in_support(self, marginals):
        """PPF output lies within the support bounds (x_lo, x_hi) for any p
        in (0, 1).  The support now extends beyond the raw strike range to
        allow the quadrature grid to cover deep-OTM constraints."""
        mX, _, _ = marginals
        x_lo, x_hi = mX.support()
        u_test = np.linspace(0.001, 0.999, 20)
        ppf_vals = mX.ppf(u_test)
        assert np.all(ppf_vals >= x_lo - 1e-10), (
            f"PPF below support low {x_lo:.6f}; got {ppf_vals.min():.6f}"
        )
        assert np.all(ppf_vals <= x_hi + 1e-10), (
            f"PPF above support high {x_hi:.6f}; got {ppf_vals.max():.6f}"
        )

    def test_support_extends_beyond_strike_range(self, marginals):
        """support() bounds extend beyond [K_min, K_max] so the quadrature
        grid has nodes past the deepest OTM market strike."""
        mX, _, data = marginals
        K = data["K_X"]
        x_lo, x_hi = mX.support()
        assert x_lo <= K[0] + 1e-10, (
            f"Support left {x_lo:.6f} should be ≤ K_min {K[0]:.6f}"
        )
        assert x_hi >= K[-1] - 1e-10, (
            f"Support right {x_hi:.6f} should be ≥ K_max {K[-1]:.6f}"
        )


# ---------------------------------------------------------------------------
# Cross-checks between CDF and PPF
# ---------------------------------------------------------------------------

class TestCDFPPFConsistency:

    def test_cdf_ppf_at_median(self, marginals):
        """cdf(ppf(0.5)) ≈ 0.5 (median)."""
        mX, _, _ = marginals
        median = mX.ppf(0.5)
        assert abs(mX.cdf(median) - 0.5) < 0.02

    def test_forward_near_median(self, marginals):
        """For a roughly symmetric distribution, cdf(F_X) ≈ 0.5."""
        mX, _, data = marginals
        cdf_fwd = mX.cdf(data["F_X"])
        # The Bachelier model is symmetric around F, so CDF(F) ≈ 0.5
        assert abs(cdf_fwd - 0.5) < 0.15
