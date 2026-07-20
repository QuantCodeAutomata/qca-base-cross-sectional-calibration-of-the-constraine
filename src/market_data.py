"""Synthetic CMS option market data generation.

Produces realistic CMS rate option data via the Bachelier (normal) model
with a quadratic implied-vol smile parameterisation.  Real CMS data is
unavailable, so all market inputs are synthesised from calibrated
scenario parameters.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Standalone Bachelier helpers (also used by reporting.py)
# ---------------------------------------------------------------------------

def bachelier_call_price(F: float, K: float | np.ndarray,
                         sigma: float | np.ndarray,
                         T: float) -> float | np.ndarray:
    """Bachelier (normal) call price.

    Parameters
    ----------
    F : float
        Forward rate.
    K : float or array
        Strike(s).
    sigma : float or array
        Annualised Bachelier vol (same shape as K, or scalar).
    T : float
        Expiry in years.

    Returns
    -------
    float or ndarray
        Call price(s)  max(F - K, 0)  under the normal model.
    """
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrtT = np.sqrt(T)
    total_vol = sigma * sqrtT

    # ATM case handled via where to avoid 0/0
    atm_mask = np.abs(K - F) < 1e-14

    # Away-from-ATM
    d = np.where(atm_mask, 0.0, (F - K) / np.where(total_vol > 0, total_vol, 1.0))
    price_otm = (F - K) * norm.cdf(d) + total_vol * norm.pdf(d)
    price_atm = total_vol / np.sqrt(2.0 * np.pi)

    price = np.where(atm_mask, price_atm, price_otm)
    return float(price) if price.ndim == 0 else price


def bachelier_implied_vol(price: float, F: float, K: float,
                          T: float, tol: float = 1e-10) -> float:
    """Invert a Bachelier call price to recover the annualised implied vol.

    Parameters
    ----------
    price : float
        Observed call price.
    F : float
        Forward rate.
    K : float
        Strike.
    T : float
        Expiry in years.
    tol : float
        Root-finding tolerance.

    Returns
    -------
    float
        Annualised Bachelier implied vol.

    Raises
    ------
    ValueError
        If the price is below intrinsic value or the root cannot be bracketed.
    """
    intrinsic = max(F - K, 0.0)
    if price < intrinsic - tol:
        raise ValueError(
            f"Price {price:.6g} is below intrinsic value {intrinsic:.6g}."
        )
    if price <= intrinsic + tol:
        return 0.0

    sqrtT = np.sqrt(T)

    def objective(sigma: float) -> float:
        return bachelier_call_price(F, K, sigma, T) - price

    # Bracket the root
    sigma_low = 1e-8
    sigma_high = 1.0  # 100% vol is extreme for rates
    while objective(sigma_high) < 0:
        sigma_high *= 2.0
        if sigma_high > 1e6:
            raise ValueError("Cannot bracket implied vol root: price too large.")

    sigma_impl = optimize.brentq(objective, sigma_low, sigma_high,
                                 xtol=tol, rtol=tol)
    return float(sigma_impl)


# ---------------------------------------------------------------------------
# Scenario parameter tables
# ---------------------------------------------------------------------------

_SCENARIO_PARAMS = {
    # CMS 20y - CMS 2y spread (payer)
    "2s20s": dict(
        F_X=0.025, F_Y=0.015,            # X = CMS20y, Y = CMS2y, Z = spread
        sigma_X_atm=0.007, sigma_Y_atm=0.005, sigma_Z_atm=0.004,
        skew_X=-0.15,  skew_Y=-0.10,
        convexity_X=0.05, convexity_Y=0.04,
        skew_Z=-0.08,  convexity_Z=0.03,
    ),
    # CMS 30y - CMS 2y spread, stressed environment
    "30y2y": dict(
        F_X=0.030, F_Y=0.015,
        sigma_X_atm=0.009, sigma_Y_atm=0.005, sigma_Z_atm=0.008,
        skew_X=-0.20,  skew_Y=-0.10,
        convexity_X=0.06, convexity_Y=0.04,
        skew_Z=-0.12,  convexity_Z=0.05,
    ),
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SyntheticCMSMarket:
    """Generator of synthetic CMS rate option market data.

    Parameters
    ----------
    scenario : str
        One of ``'2s20s'`` (CMS20y/CMS2y) or ``'30y2y'`` (CMS30y/CMS2y).
    expiry_years : float
        Option expiry expressed in years (default 5.0).
    stressed : bool
        When *True*, override scenario parameters with stressed magnitudes.
        Currently the ``'30y2y'`` scenario is itself the stressed scenario;
        this flag is reserved for future use.
    """

    def __init__(self,
                 scenario: str = "2s20s",
                 expiry_years: float = 5.0,
                 stressed: bool = False) -> None:
        if scenario not in _SCENARIO_PARAMS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Choose from {list(_SCENARIO_PARAMS)}."
            )
        self.scenario = scenario
        self.T = expiry_years
        self.stressed = stressed

        params = dict(_SCENARIO_PARAMS[scenario])
        # stressed flag: amplify vols and skews by 20 %
        if stressed:
            for key in ("sigma_X_atm", "sigma_Y_atm", "sigma_Z_atm"):
                params[key] *= 1.2
            for key in ("skew_X", "skew_Y", "skew_Z"):
                params[key] *= 1.2

        self.F_X: float = params["F_X"]
        self.F_Y: float = params["F_Y"]
        self.F_Z: float = self.F_X - self.F_Y          # spread forward

        self.sigma_X_atm: float = params["sigma_X_atm"]
        self.sigma_Y_atm: float = params["sigma_Y_atm"]
        self.sigma_Z_atm: float = params["sigma_Z_atm"]

        self.skew_X: float      = params["skew_X"]
        self.skew_Y: float      = params["skew_Y"]
        self.skew_Z: float      = params["skew_Z"]
        self.convexity_X: float = params["convexity_X"]
        self.convexity_Y: float = params["convexity_Y"]
        self.convexity_Z: float = params["convexity_Z"]

    # ------------------------------------------------------------------
    # Vol smile
    # ------------------------------------------------------------------

    def bachelier_vol(self, strike: float | np.ndarray,
                      forward: float,
                      sigma_atm: float,
                      skew: float,
                      convexity: float) -> float | np.ndarray:
        """Quadratic Bachelier implied-vol smile.

        sigma(K) = sigma_atm * (1 + skew*(K-F) + convexity*(K-F)^2),
        clipped to max(1e-4, result).

        Parameters
        ----------
        strike : float or array
            Strike rate(s).
        forward : float
            Forward rate.
        sigma_atm : float
            At-the-money annualised Bachelier vol.
        skew : float
            Linear skew coefficient.
        convexity : float
            Quadratic convexity coefficient.

        Returns
        -------
        float or ndarray
            Implied vol(s), clipped below at 1e-4.
        """
        dK = np.asarray(strike, dtype=float) - forward
        vol = sigma_atm * (1.0 + skew * dK + convexity * dK ** 2)
        return np.clip(vol, 1e-4, None)

    # ------------------------------------------------------------------
    # Bachelier call pricer (instance method for convenience)
    # ------------------------------------------------------------------

    def bachelier_call(self, forward: float, strike: float | np.ndarray,
                       sigma: float | np.ndarray,
                       T: float) -> float | np.ndarray:
        """Bachelier call price — delegates to the module-level function."""
        return bachelier_call_price(forward, strike, sigma, T)

    # ------------------------------------------------------------------
    # Strike grid
    # ------------------------------------------------------------------

    def get_strikes(self, forward: float, sigma_atm: float,
                    n_strikes: int = 20, n_std: int = 3) -> np.ndarray:
        """Linearly spaced strike grid spanning ±n_std total-vol around F.

        Parameters
        ----------
        forward : float
            Forward rate.
        sigma_atm : float
            ATM annualised Bachelier vol.
        n_strikes : int
            Number of strikes (default 20).
        n_std : int
            Number of standard deviations on each side.

        Returns
        -------
        ndarray of shape (n_strikes,)
        """
        half_width = n_std * sigma_atm * np.sqrt(self.T)
        return np.linspace(forward - half_width, forward + half_width, n_strikes)

    # ------------------------------------------------------------------
    # Market data bundle
    # ------------------------------------------------------------------

    def get_market_data(self) -> dict:
        """Assemble the full synthetic market data bundle.

        Returns
        -------
        dict with keys
            ``F_X``, ``F_Y``, ``F_Z`` : float — forward rates
            ``T``                      : float — expiry in years
            ``K_X``, ``K_Y``, ``K_Z`` : ndarray — strike grids
            ``C_X``, ``C_Y``, ``C_Z`` : ndarray — Bachelier call prices
            ``sigma_X``, ``sigma_Y``, ``sigma_Z`` : float — ATM total vols
              (= sigma_atm * sqrt(T)) used as scale references
        """
        T = self.T

        # Strike grids
        K_X = self.get_strikes(self.F_X, self.sigma_X_atm)
        K_Y = self.get_strikes(self.F_Y, self.sigma_Y_atm)
        K_Z = self.get_strikes(self.F_Z, self.sigma_Z_atm)

        # Per-strike Bachelier vols (annualised)
        vol_X = self.bachelier_vol(K_X, self.F_X,
                                   self.sigma_X_atm, self.skew_X, self.convexity_X)
        vol_Y = self.bachelier_vol(K_Y, self.F_Y,
                                   self.sigma_Y_atm, self.skew_Y, self.convexity_Y)
        vol_Z = self.bachelier_vol(K_Z, self.F_Z,
                                   self.sigma_Z_atm, self.skew_Z, self.convexity_Z)

        # Call prices
        C_X = bachelier_call_price(self.F_X, K_X, vol_X, T)
        C_Y = bachelier_call_price(self.F_Y, K_Y, vol_Y, T)
        C_Z = bachelier_call_price(self.F_Z, K_Z, vol_Z, T)

        # ATM total vols (sigma_atm * sqrt(T)) — scale references
        sigma_X = self.sigma_X_atm * np.sqrt(T)
        sigma_Y = self.sigma_Y_atm * np.sqrt(T)
        sigma_Z = self.sigma_Z_atm * np.sqrt(T)

        return {
            "F_X": self.F_X,  "F_Y": self.F_Y,  "F_Z": self.F_Z,
            "T":   T,
            "K_X": K_X,       "K_Y": K_Y,       "K_Z": K_Z,
            "C_X": C_X,       "C_Y": C_Y,       "C_Z": C_Z,
            "sigma_X": sigma_X, "sigma_Y": sigma_Y, "sigma_Z": sigma_Z,
        }
