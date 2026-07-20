"""
Tests for src/reporting.py.

Covers:
- prices_to_bachelier_vols: non-negativity, price roundtrip
- compute_calibration_diagnostics: structure check
- gaussian_copula_price: positivity, zero-strike limit
- compute_implied_gaussian_correlation: roundtrip
- plot functions: return Matplotlib figures
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from src.market_data import SyntheticCMSMarket, bachelier_call_price
from src.marginals import MarginalDistribution
from src.quadrature import build_quadrature_grid
from src.copulas import build_prior_weights
from src.reporting import (
    prices_to_bachelier_vols,
    compute_calibration_diagnostics,
    gaussian_copula_price,
    compute_implied_gaussian_correlation,
    plot_smile_comparison,
    plot_joint_distribution,
)


# ---------------------------------------------------------------------------
# Module-level fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reporting_setup():
    """Build market data, marginals, grid, Q, and W_ij for reporting tests."""
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
    Q_ij = build_prior_weights(mX, mY, grid, copula_type="gaussian", rho=0.78)

    return dict(data=data, mX=mX, mY=mY, grid=grid, Q_ij=Q_ij)


# ---------------------------------------------------------------------------
# prices_to_bachelier_vols
# ---------------------------------------------------------------------------

class TestPricesToBachelierVols:

    def test_prices_to_vols_positive(self, reporting_setup):
        """Implied vols are non-negative for all valid call prices."""
        d = reporting_setup
        vols = prices_to_bachelier_vols(
            d["data"]["C_X"], d["data"]["F_X"], d["data"]["K_X"], d["data"]["T"]
        )
        assert np.all(vols >= 0.0), f"Negative vols found: {vols.min():.3e}"

    def test_prices_to_vols_roundtrip(self, reporting_setup):
        """prices → vols → prices roundtrip error < 1e-8."""
        d = reporting_setup
        prices_in = d["data"]["C_X"]
        F, K, T = d["data"]["F_X"], d["data"]["K_X"], d["data"]["T"]

        vols = prices_to_bachelier_vols(prices_in, F, K, T)
        prices_out = bachelier_call_price(F, K, vols, T)

        max_err = np.max(np.abs(prices_out - prices_in))
        assert max_err < 1e-8, f"Roundtrip error {max_err:.3e} >= 1e-8"

    def test_prices_to_vols_for_Y(self, reporting_setup):
        """Roundtrip also works for the Y market."""
        d = reporting_setup
        prices_in = d["data"]["C_Y"]
        F, K, T = d["data"]["F_Y"], d["data"]["K_Y"], d["data"]["T"]

        vols = prices_to_bachelier_vols(prices_in, F, K, T)
        prices_out = bachelier_call_price(F, K, vols, T)

        assert np.max(np.abs(prices_out - prices_in)) < 1e-8

    def test_prices_to_vols_atm_close_to_market_vol(self, reporting_setup):
        """ATM implied vol agrees with market ATM vol to within 1 bps."""
        d = reporting_setup
        mkt = SyntheticCMSMarket(scenario="2s20s", expiry_years=5.0)
        F, K, T = d["data"]["F_X"], d["data"]["K_X"], d["data"]["T"]

        # Find the ATM strike index
        atm_idx = np.argmin(np.abs(K - F))
        prices = np.array([d["data"]["C_X"][atm_idx]])
        vols = prices_to_bachelier_vols(prices, F, np.array([K[atm_idx]]), T)

        # ATM Bachelier vol ≈ market sigma_X_atm
        sigma_atm = mkt.sigma_X_atm
        assert abs(vols[0] - sigma_atm) < 1e-4  # within 1 bps = 0.0001


# ---------------------------------------------------------------------------
# compute_calibration_diagnostics
# ---------------------------------------------------------------------------

class TestCalibrationDiagnostics:

    def test_calibration_diagnostics_structure(self, reporting_setup):
        """compute_calibration_diagnostics returns all expected dict keys."""
        d = reporting_setup
        diag = compute_calibration_diagnostics(
            d["Q_ij"], d["grid"]["x_nodes"], d["grid"]["y_nodes"],
            d["data"], epsilon=1.0,
        )
        expected_keys = {
            "normalization_error",
            "X_moment_error",
            "Y_moment_error",
            "X_price_errors",
            "Y_price_errors",
            "Z_price_errors",
            "X_vol_errors",
            "Y_vol_errors",
            "Z_vol_errors",
            "max_abs_price_error",
            "max_abs_vol_error_bps",
        }
        missing = expected_keys - set(diag.keys())
        assert not missing, f"Missing keys in diagnostics: {missing}"

    def test_calibration_diagnostics_normalization_prior(self, reporting_setup):
        """Normalisation error is 0 for a properly normalised Q_ij."""
        d = reporting_setup
        diag = compute_calibration_diagnostics(
            d["Q_ij"], d["grid"]["x_nodes"], d["grid"]["y_nodes"],
            d["data"], epsilon=1.0,
        )
        assert abs(diag["normalization_error"]) < 1e-12

    def test_calibration_diagnostics_vol_errors_array(self, reporting_setup):
        """X/Y/Z vol_errors entries are arrays matching the strike count."""
        d = reporting_setup
        diag = compute_calibration_diagnostics(
            d["Q_ij"], d["grid"]["x_nodes"], d["grid"]["y_nodes"],
            d["data"], epsilon=1.0,
        )
        assert len(diag["X_vol_errors"]) == len(d["data"]["K_X"])
        assert len(diag["Y_vol_errors"]) == len(d["data"]["K_Y"])
        assert len(diag["Z_vol_errors"]) == len(d["data"]["K_Z"])

    def test_calibration_diagnostics_max_errors_nonneg(self, reporting_setup):
        """max_abs_price_error and max_abs_vol_error_bps are non-negative."""
        d = reporting_setup
        diag = compute_calibration_diagnostics(
            d["Q_ij"], d["grid"]["x_nodes"], d["grid"]["y_nodes"],
            d["data"], epsilon=1.0,
        )
        assert diag["max_abs_price_error"] >= 0.0
        assert diag["max_abs_vol_error_bps"] >= 0.0


# ---------------------------------------------------------------------------
# gaussian_copula_price
# ---------------------------------------------------------------------------

class TestGaussianCopulaPrice:

    def test_gaussian_copula_price_positive(self, reporting_setup):
        """Call on X + Y has a non-negative price for any valid rho."""
        d = reporting_setup
        K = d["data"]["F_X"] + d["data"]["F_Y"]  # ATM for the sum
        for rho in [-0.5, 0.0, 0.5, 0.9]:
            price = gaussian_copula_price(
                d["mX"], d["mY"], rho_impl=rho, strike=K, quad_grid=d["grid"]
            )
            assert price >= 0.0, f"Negative price {price:.6f} for rho={rho}"

    def test_gaussian_copula_price_zero_strike(self, reporting_setup):
        """Price at K=0 approximates E[X] + E[Y] = F_X + F_Y (within 10 %)."""
        d = reporting_setup
        # max(X+Y-0, 0) ≈ X+Y when X+Y is rarely negative
        price = gaussian_copula_price(
            d["mX"], d["mY"], rho_impl=0.5, strike=0.0, quad_grid=d["grid"]
        )
        expected = d["data"]["F_X"] + d["data"]["F_Y"]
        rel_err = abs(price - expected) / expected
        assert rel_err < 0.10, (
            f"price={price:.4f}, E[X+Y]≈{expected:.4f}, rel_err={rel_err:.3f}"
        )

    def test_gaussian_copula_price_increases_with_rho(self, reporting_setup):
        """Higher rho → higher call price at ATM (positive dependence)."""
        d = reporting_setup
        K = d["data"]["F_X"] + d["data"]["F_Y"]
        p_lo = gaussian_copula_price(d["mX"], d["mY"], rho_impl=-0.5,
                                     strike=K, quad_grid=d["grid"])
        p_hi = gaussian_copula_price(d["mX"], d["mY"], rho_impl=0.9,
                                     strike=K, quad_grid=d["grid"])
        assert p_hi > p_lo, (
            "Higher rho should give a higher call-on-sum price"
        )

    def test_gaussian_copula_price_deep_otm_near_zero(self, reporting_setup):
        """Deep-OTM price is small (well below F_X + F_Y)."""
        d = reporting_setup
        # Strike much larger than the support of X+Y
        K_otm = d["data"]["F_X"] + d["data"]["F_Y"] + 0.15
        price = gaussian_copula_price(
            d["mX"], d["mY"], rho_impl=0.5, strike=K_otm, quad_grid=d["grid"]
        )
        assert price < 1e-6, f"Deep-OTM price too large: {price:.3e}"


# ---------------------------------------------------------------------------
# compute_implied_gaussian_correlation
# ---------------------------------------------------------------------------

class TestImpliedGaussianCorrelation:

    def test_implied_correlation_roundtrip(self, reporting_setup):
        """price → implied rho → price recovers original price."""
        d = reporting_setup
        K = d["data"]["F_X"] + d["data"]["F_Y"]

        # Price a call at rho=0.5
        rho_in = 0.5
        price_in = gaussian_copula_price(
            d["mX"], d["mY"], rho_impl=rho_in, strike=K, quad_grid=d["grid"]
        )

        # Invert to get implied rho
        rho_impl = compute_implied_gaussian_correlation(
            price_in, d["mX"], d["mY"], K, d["grid"]
        )

        # Check closeness to original rho
        assert abs(rho_impl - rho_in) < 1e-4, (
            f"rho roundtrip: in={rho_in}, implied={rho_impl:.6f}"
        )

    def test_implied_correlation_in_bounds(self, reporting_setup):
        """Implied correlation lies in (−1, 1)."""
        d = reporting_setup
        K = d["data"]["F_X"] + d["data"]["F_Y"]
        price = gaussian_copula_price(
            d["mX"], d["mY"], rho_impl=0.5, strike=K, quad_grid=d["grid"]
        )
        rho_impl = compute_implied_gaussian_correlation(
            price, d["mX"], d["mY"], K, d["grid"]
        )
        assert -1.0 < rho_impl < 1.0

    def test_implied_correlation_out_of_range_raises(self, reporting_setup):
        """ValueError raised if the price is outside the achievable range."""
        d = reporting_setup
        K = d["data"]["F_X"] + d["data"]["F_Y"]
        # Price that is unreachably large (larger than upper-bound price)
        with pytest.raises(ValueError):
            compute_implied_gaussian_correlation(
                999.0, d["mX"], d["mY"], K, d["grid"]
            )


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

class TestPlotFunctions:

    def test_plot_smile_comparison_returns_figure(self, reporting_setup):
        """plot_smile_comparison returns a Matplotlib Figure."""
        import matplotlib.pyplot as plt
        d = reporting_setup
        fig = plot_smile_comparison(
            d["data"], d["Q_ij"],
            d["grid"]["x_nodes"], d["grid"]["y_nodes"],
            title_suffix="Test",
            save_path=None,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_joint_distribution_returns_figure(self, reporting_setup):
        """plot_joint_distribution returns a Matplotlib Figure."""
        import matplotlib.pyplot as plt
        d = reporting_setup
        fig = plot_joint_distribution(
            d["Q_ij"], d["grid"]["x_nodes"], d["grid"]["y_nodes"],
            save_path=None,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
