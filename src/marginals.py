"""Marginal CDF/PDF extraction from Bachelier option prices.

Uses the Breeden-Litzenberger relation  f(K) = d²C/dK²  to recover the
risk-neutral density from a Bachelier call-price surface, after smoothing
the implied-vol smile with a cubic spline.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq

from .market_data import bachelier_call_price, bachelier_implied_vol


class MarginalDistribution:
    """Risk-neutral marginal distribution extracted from Bachelier call prices.

    The procedure is:
    1. Invert each (strike, call-price) pair to Bachelier implied vol.
    2. Fit a quadratic polynomial to the implied-vol smile for regularisation.
    3. Build a ``CubicSpline`` on the smoothed smile and evaluate it on a
       fine grid of 3 000 points spanning
       ``[K_min - 2*(K_max-K_min), K_max + 2*(K_max-K_min)]``, with vol
       held flat (constant boundary value) outside ``[K_min, K_max]``.
    4. Compute Bachelier call prices on the fine grid.
    5. Apply Breeden-Litzenberger (second finite-difference of calls) to get
       the risk-neutral density.
    6. Clip, normalise, and integrate to obtain a valid CDF.

    Parameters
    ----------
    strikes : array-like
        Market strike grid (increasing).
    call_prices : array-like
        Market Bachelier call prices corresponding to *strikes*.
    forward : float
        Forward rate for this marginal.
    T : float
        Option expiry in years.
    q_trunc : float
        Truncation probability for the support (default 1e-4).
    """

    def __init__(self,
                 strikes: np.ndarray,
                 call_prices: np.ndarray,
                 forward: float,
                 T: float,
                 q_trunc: float = 1e-4) -> None:
        self.strikes     = np.asarray(strikes,     dtype=float)
        self.call_prices = np.asarray(call_prices, dtype=float)
        self.forward     = float(forward)
        self.T           = float(T)
        self.q_trunc     = float(q_trunc)

        if self.strikes.ndim != 1 or len(self.strikes) < 4:
            raise ValueError("Need at least 4 strikes to build a distribution.")

        self._build_density()

    # ------------------------------------------------------------------
    # Internal construction
    # ------------------------------------------------------------------

    def _build_density(self) -> None:
        """Fit smile, apply Breeden-Litzenberger, build CDF/PDF interpolants."""
        K_in  = self.strikes
        C_in  = self.call_prices
        F     = self.forward
        T     = self.T

        # ---- step 1: compute implied vols at input strikes ---------------
        sigma_impl = np.empty_like(K_in)
        for i, (k, c) in enumerate(zip(K_in, C_in)):
            try:
                sigma_impl[i] = bachelier_implied_vol(c, F, k, T)
            except Exception:
                # Fallback: use ATM-like value from neighbours (filled later)
                sigma_impl[i] = np.nan

        # Fill NaNs by linear interpolation from valid neighbours
        valid = np.isfinite(sigma_impl) & (sigma_impl > 0)
        if valid.sum() < 4:
            raise RuntimeError(
                "Too few valid implied vols to fit the smile.  "
                "Check input call prices."
            )
        if not np.all(valid):
            xp = K_in[valid]
            fp = sigma_impl[valid]
            sigma_impl = np.interp(K_in, xp, fp)

        # ---- step 2: quadratic polynomial regularisation -----------------
        poly_coeffs  = np.polyfit(K_in, sigma_impl, 2)
        sigma_smooth = np.polyval(poly_coeffs, K_in)
        sigma_smooth = np.clip(sigma_smooth, 1e-4, None)

        # ---- step 3: CubicSpline on smoothed smile ----------------------
        cs = CubicSpline(K_in, sigma_smooth, bc_type="not-a-knot")

        # Fine grid – 3 000 points spanning 5× the original strike range for
        # better tail coverage; vol is held flat outside [K_in[0], K_in[-1]].
        K_range = K_in[-1] - K_in[0]
        K_fine = np.linspace(K_in[0] - 2.0 * K_range,
                             K_in[-1] + 2.0 * K_range, 3000)
        self.K_fine = K_fine

        sigma_fine = cs(np.clip(K_fine, K_in[0], K_in[-1]))
        sigma_fine = np.clip(sigma_fine, 1e-4, None)

        # ---- step 4: Bachelier call prices on fine grid ------------------
        C_fine = bachelier_call_price(F, K_fine, sigma_fine, T)

        # ---- step 5: Breeden-Litzenberger  f = d²C/dK² ------------------
        dK = K_fine[1] - K_fine[0]       # uniform spacing
        # second-order central differences; boundary via one-sided FD
        density = np.empty_like(C_fine)
        density[1:-1] = (C_fine[2:] - 2.0 * C_fine[1:-1] + C_fine[:-2]) / dK ** 2
        density[0]    = density[1]        # forward difference boundary
        density[-1]   = density[-2]

        # ---- step 6: clip and normalise ----------------------------------
        density = np.clip(density, 0.0, None)
        mass    = np.trapz(density, K_fine)
        if mass <= 0.0:
            raise RuntimeError("Density integrates to zero — check inputs.")
        density /= mass

        # ---- step 7: CDF via cumulative trapezoid ------------------------
        cdf_vals = np.concatenate(([0.0],
                                   cumulative_trapezoid(density, K_fine)))
        cdf_vals = np.clip(cdf_vals, 0.0, 1.0)

        # Ensure strict monotonicity required by brentq / interp
        eps_mono = np.finfo(float).eps
        for i in range(1, len(cdf_vals)):
            if cdf_vals[i] <= cdf_vals[i - 1]:
                cdf_vals[i] = cdf_vals[i - 1] + eps_mono

        # ---- step 8: build interpolants ----------------------------------
        self._cdf_interp = interp1d(
            K_fine, cdf_vals,
            kind="linear",
            bounds_error=False,
            fill_value=(0.0, 1.0),
        )
        self._pdf_interp = interp1d(
            K_fine, density,
            kind="linear",
            bounds_error=False,
            fill_value=(0.0, 0.0),
        )

        # ppf search bounds: use the full extended K_fine range so the
        # q_trunc quantile can be resolved well beyond the market-strike range.
        # This prevents the quadrature support from being clamped to [K_in[0],
        # K_in[-1]], which would place the deepest OTM strike at the grid
        # boundary and make its call-price constraint permanently unsatisfiable.
        self._x_low  = float(K_fine[0])
        self._x_high = float(K_fine[-1])

        # CDF values at the ppf search boundaries (precomputed for efficiency).
        self._cdf_at_x_low  = float(self._cdf_interp(self._x_low))
        self._cdf_at_x_high = float(self._cdf_interp(self._x_high))

        # Refine support via ppf
        q = self.q_trunc
        self._support_low  = self._ppf_raw(q)
        self._support_high = self._ppf_raw(1.0 - q)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Cumulative distribution function M(x), clipped to [0, 1].

        Parameters
        ----------
        x : float or array
            Evaluation point(s).

        Returns
        -------
        float or ndarray
        """
        result = np.clip(self._cdf_interp(x), 0.0, 1.0)
        return float(result) if np.ndim(x) == 0 else result

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Probability density function f(x), clipped to >= 0.

        Parameters
        ----------
        x : float or array
            Evaluation point(s).

        Returns
        -------
        float or ndarray
        """
        result = np.clip(self._pdf_interp(x), 0.0, None)
        return float(result) if np.ndim(x) == 0 else result

    def _ppf_raw(self, p: float) -> float:
        """Inverse CDF without truncation guard."""
        p = float(np.clip(p, 0.0, 1.0))
        if p <= 0.0:
            return self._x_low
        if p >= 1.0:
            return self._x_high
        # The extended K_fine grid gives CDF > 0 at _x_low and CDF < 1 at
        # _x_high.  Clamp p to the achievable CDF range so brentq always
        # has a valid bracket within [_x_low, _x_high].
        if p <= self._cdf_at_x_low:
            return self._x_low
        if p >= self._cdf_at_x_high:
            return self._x_high
        try:
            return brentq(
                lambda x: float(self._cdf_interp(x)) - p,
                self._x_low, self._x_high,
                xtol=1e-10, rtol=1e-10,
            )
        except ValueError:
            # Fallback: linear interpolation into cdf values, clamped to
            # [_x_low, _x_high] to stay within the market-data strike range.
            cdf_vals = self._cdf_interp(self.K_fine)
            idx = np.searchsorted(cdf_vals, p)
            idx = np.clip(idx, 0, len(self.K_fine) - 1)
            return float(np.clip(self.K_fine[idx], self._x_low, self._x_high))

    def ppf(self, p: float | np.ndarray) -> float | np.ndarray:
        """Percent-point function (inverse CDF).

        Parameters
        ----------
        p : float or array
            Probability level(s) in [0, 1].

        Returns
        -------
        float or ndarray
        """
        scalar = np.ndim(p) == 0
        p_arr  = np.atleast_1d(np.asarray(p, dtype=float))
        out    = np.empty_like(p_arr)

        q = self.q_trunc
        for i, pi in enumerate(p_arr.flat):
            if pi <= q:
                out.flat[i] = self._support_low
            elif pi >= 1.0 - q:
                out.flat[i] = self._support_high
            else:
                out.flat[i] = self._ppf_raw(pi)

        return float(out.flat[0]) if scalar else out

    def support(self) -> tuple[float, float]:
        """Return the effective support ``(x_low, x_high)``.

        The bounds correspond to the *q_trunc* and *(1 - q_trunc)* quantiles
        of the density.

        Returns
        -------
        tuple (x_low, x_high)
        """
        return (self._support_low, self._support_high)
