"""
Tests for src/market_data.py.

Covers:
- Bachelier call pricing
- Put-call parity
- Implied-vol roundtrip
- SyntheticCMSMarket: forward ordering, strike count, price positivity,
  and monotone decreasing prices.
"""
import numpy as np
import pytest
from scipy.stats import norm

from src.market_data import bachelier_call_price, bachelier_implied_vol, SyntheticCMSMarket


# ---------------------------------------------------------------------------
# Bachelier pricing
# ---------------------------------------------------------------------------

class TestBachelierCallPrice:

    def test_bachelier_call_positive(self):
        """Call price is non-negative for all input combinations."""
        F, sigma, T = 0.04, 0.008, 5.0
        for K in [0.02, 0.04, 0.06]:
            price = bachelier_call_price(F, K, sigma, T)
            assert price >= 0.0, f"Negative call price for K={K}"

    def test_bachelier_call_atm(self):
        """ATM Bachelier call equals sigma*sqrt(T) / sqrt(2π)."""
        F, K, sigma, T = 0.025, 0.025, 0.007, 5.0
        price = bachelier_call_price(F, K, sigma, T)
        expected = sigma * np.sqrt(T) / np.sqrt(2.0 * np.pi)
        assert abs(price - expected) < 1e-12, (
            f"ATM price {price:.8f} ≠ expected {expected:.8f}"
        )

    def test_bachelier_call_put_parity(self):
        """Put-call parity: C − P = F − K under Bachelier model."""
        F, K, sigma, T = 0.025, 0.030, 0.007, 5.0
        # Call price from module
        C = bachelier_call_price(F, K, sigma, T)
        # Put price from Bachelier formula  P = (K-F)*Φ(-d) + σ√T φ(d)
        d = (F - K) / (sigma * np.sqrt(T))
        P = (K - F) * norm.cdf(-d) + sigma * np.sqrt(T) * norm.pdf(d)
        # Put-call parity
        assert abs(C - P - (F - K)) < 1e-10, (
            f"Parity violated: C-P={C-P:.8f}, F-K={F-K:.8f}"
        )

    def test_bachelier_call_itm_approaches_intrinsic(self):
        """Deep-ITM call approaches intrinsic value F - K."""
        F, K, sigma, T = 0.04, 0.005, 0.007, 5.0  # K much less than F
        price = bachelier_call_price(F, K, sigma, T)
        intrinsic = F - K
        # Price must be >= intrinsic and close to it for deep ITM
        assert price >= intrinsic - 1e-10
        # For very deep ITM, the optionality premium is small
        assert price < intrinsic + 0.01  # within 100 bps of intrinsic

    def test_bachelier_call_otm_approaches_zero(self):
        """Deep-OTM call price approaches zero."""
        F, K, sigma, T = 0.025, 0.10, 0.007, 5.0  # K >> F
        price = bachelier_call_price(F, K, sigma, T)
        assert price < 1e-4, f"Deep-OTM call price too large: {price}"

    def test_bachelier_call_array_input(self):
        """bachelier_call_price handles array K inputs."""
        F, sigma, T = 0.025, 0.007, 5.0
        K = np.linspace(0.01, 0.05, 10)
        prices = bachelier_call_price(F, K, sigma, T)
        assert prices.shape == (10,)
        assert np.all(prices >= 0)


# ---------------------------------------------------------------------------
# Implied vol inversion
# ---------------------------------------------------------------------------

class TestBachelierImpliedVol:

    def test_bachelier_implied_vol_roundtrip(self):
        """price → implied vol → price recovers original price."""
        F, K, sigma, T = 0.025, 0.030, 0.008, 5.0
        price_in = bachelier_call_price(F, K, sigma, T)
        vol_impl = bachelier_implied_vol(price_in, F, K, T)
        price_out = bachelier_call_price(F, K, vol_impl, T)
        assert abs(price_out - price_in) < 1e-8, (
            f"Roundtrip error: {abs(price_out - price_in):.2e}"
        )

    def test_bachelier_implied_vol_atm(self):
        """Implied vol at ATM recovers the original ATM vol."""
        F, sigma, T = 0.025, 0.007, 5.0
        K = F  # ATM
        price = bachelier_call_price(F, K, sigma, T)
        vol_impl = bachelier_implied_vol(price, F, K, T)
        assert abs(vol_impl - sigma) < 1e-8

    def test_bachelier_implied_vol_otm(self):
        """Implied vol for OTM call roundtrips correctly."""
        F, K, sigma, T = 0.025, 0.035, 0.007, 5.0
        price = bachelier_call_price(F, K, sigma, T)
        vol_impl = bachelier_implied_vol(price, F, K, T)
        assert abs(vol_impl - sigma) < 1e-7

    def test_bachelier_implied_vol_positive(self):
        """Implied vol is non-negative."""
        F, K, sigma, T = 0.025, 0.028, 0.007, 5.0
        price = bachelier_call_price(F, K, sigma, T)
        vol_impl = bachelier_implied_vol(price, F, K, T)
        assert vol_impl >= 0.0

    def test_bachelier_implied_vol_below_intrinsic_raises(self):
        """ValueError raised when price is below intrinsic."""
        F, K, T = 0.04, 0.02, 5.0  # ITM, intrinsic = 0.02
        bad_price = 0.01  # below intrinsic
        with pytest.raises(ValueError):
            bachelier_implied_vol(bad_price, F, K, T)


# ---------------------------------------------------------------------------
# SyntheticCMSMarket
# ---------------------------------------------------------------------------

class TestSyntheticCMSMarket:

    @pytest.fixture(autouse=True)
    def _market_data(self):
        mkt = SyntheticCMSMarket(scenario="2s20s", expiry_years=5.0)
        self.data = mkt.get_market_data()

    def test_synthetic_cms_market_2s20s(self):
        """2s20s: F_X > F_Y, and F_Z = F_X − F_Y."""
        d = self.data
        assert d["F_X"] > d["F_Y"], "Expected F_X > F_Y for 2s20s"
        assert abs(d["F_Z"] - (d["F_X"] - d["F_Y"])) < 1e-12

    def test_strike_grid_length(self):
        """Strike grid has exactly 20 strikes per asset."""
        d = self.data
        assert len(d["K_X"]) == 20
        assert len(d["K_Y"]) == 20
        assert len(d["K_Z"]) == 20

    def test_call_prices_positive(self):
        """All generated call prices are non-negative."""
        d = self.data
        assert np.all(d["C_X"] >= 0), "Negative C_X prices"
        assert np.all(d["C_Y"] >= 0), "Negative C_Y prices"
        assert np.all(d["C_Z"] >= 0), "Negative C_Z prices"

    def test_call_prices_decreasing(self):
        """Call prices are non-increasing as strike increases."""
        d = self.data
        tol = 1e-10  # allow for tiny floating-point noise
        assert np.all(np.diff(d["C_X"]) <= tol), "C_X is not non-increasing in K"
        assert np.all(np.diff(d["C_Y"]) <= tol), "C_Y is not non-increasing in K"
        assert np.all(np.diff(d["C_Z"]) <= tol), "C_Z is not non-increasing in K"

    def test_forwards_consistent_with_atm_call(self):
        """ATM Bachelier call price formula holds exactly at K = F."""
        mkt = SyntheticCMSMarket(scenario="2s20s", expiry_years=5.0)
        F_X = mkt.F_X
        T = mkt.T
        sigma_atm = mkt.sigma_X_atm
        # Evaluate the Bachelier formula directly at K = F (true ATM)
        price_atm = bachelier_call_price(F_X, F_X, sigma_atm, T)
        expected = sigma_atm * np.sqrt(T) / np.sqrt(2.0 * np.pi)
        # This is an exact identity for the Bachelier ATM formula
        assert abs(price_atm - expected) < 1e-12

    def test_strike_grids_centered_on_forward(self):
        """Strike grids are approximately centred on the forward rate."""
        d = self.data
        for K_arr, F in [(d["K_X"], d["F_X"]),
                         (d["K_Y"], d["F_Y"]),
                         (d["K_Z"], d["F_Z"])]:
            centre = 0.5 * (K_arr[0] + K_arr[-1])
            assert abs(centre - F) < abs(K_arr[-1] - K_arr[0]) * 0.01

    def test_stressed_scenario_higher_vols(self):
        """Stressed scenario returns higher ATM vols than normal."""
        mkt_normal = SyntheticCMSMarket(scenario="30y2y", expiry_years=5.0, stressed=False)
        mkt_stress = SyntheticCMSMarket(scenario="30y2y", expiry_years=5.0, stressed=True)
        assert mkt_stress.sigma_X_atm > mkt_normal.sigma_X_atm
        assert mkt_stress.sigma_Y_atm > mkt_normal.sigma_Y_atm
