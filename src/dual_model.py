"""Constrained Schrödinger optimal-transport dual model.

Implements the dual parameterisation and damped-Newton solver for the
entropic OT calibration problem with moment and call-price constraints.

The dual variable vector is
    theta = [m0, m1, m2, q_X[0..nK_X-1], q_Y[0..nK_Y-1], q_Z[0..nK_Z-1]]
and the dual potential on the quadrature grid is
    A_ij = m0 + m1*x_i + m2*y_j
         + Σ_r q_r^X * relu(x_i  - K_X_r)
         + Σ_r q_r^Y * relu(y_j  - K_Y_r)
         + Σ_r q_r^Z * relu(x_i - y_j - K_Z_r)
         - beta * Exotic_ij        (only when beta ≠ 0)

The (concave) dual objective is
    L(theta) = m0 + m1*E_X + m2*E_Y
             + q_X · C_X + q_Y · C_Y + q_Z · C_Z
             - ε * Σ_ij Q_ij * exp(A_ij / ε)  +  ε

and is maximised by a damped-Newton method with Armijo line-search.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    """Element-wise rectified linear unit  max(x, 0)."""
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Dual model
# ---------------------------------------------------------------------------

class DualModel:
    """Dual Schrödinger OT model for CMS rate calibration.

    Parameters
    ----------
    x_nodes, y_nodes : ndarray, shape (n,) and (m,)
        Quadrature nodes for X and Y.
    x_weights, y_weights : ndarray, shape (n,) and (m,)
        Quadrature weights (GL-scaled).
    Q_ij : ndarray, shape (n, m)
        Discrete prior joint weights (sum = 1).
    K_X, K_Y, K_Z : ndarray
        Strike grids for X, Y, and Z = X - Y call constraints.
    C_X_prices, C_Y_prices, C_Z_prices : ndarray
        Market Bachelier call prices corresponding to K_X, K_Y, K_Z.
    E_X, E_Y : float
        Expected values (forwards) of X and Y used in the linear
        constraints  E[X] = E_X,  E[Y] = E_Y.
    exotic_payoff_matrix : ndarray (n, m) or None
        Payoff matrix of the exotic instrument (default None → beta ignored).
    beta : float
        Lagrange multiplier / hedge ratio for the exotic instrument (default 0).
    epsilon : float
        Entropic regularisation parameter (default 1.0).
    """

    def __init__(
        self,
        x_nodes:           np.ndarray,
        y_nodes:           np.ndarray,
        x_weights:         np.ndarray,
        y_weights:         np.ndarray,
        Q_ij:              np.ndarray,
        K_X:               np.ndarray,
        K_Y:               np.ndarray,
        K_Z:               np.ndarray,
        C_X_prices:        np.ndarray,
        C_Y_prices:        np.ndarray,
        C_Z_prices:        np.ndarray,
        E_X:               float,
        E_Y:               float,
        exotic_payoff_matrix: Optional[np.ndarray] = None,
        beta:              float = 0.0,
        epsilon:           float = 1.0,
    ) -> None:
        self.x_nodes   = np.asarray(x_nodes,   dtype=float)
        self.y_nodes   = np.asarray(y_nodes,   dtype=float)
        self.x_weights = np.asarray(x_weights, dtype=float)
        self.y_weights = np.asarray(y_weights, dtype=float)
        self._Q_ij     = np.asarray(Q_ij,      dtype=float)
        self._log_Q    = np.log(self._Q_ij + 1e-300)

        self.K_X = np.asarray(K_X, dtype=float)
        self.K_Y = np.asarray(K_Y, dtype=float)
        self.K_Z = np.asarray(K_Z, dtype=float)
        self.C_X = np.asarray(C_X_prices, dtype=float)
        self.C_Y = np.asarray(C_Y_prices, dtype=float)
        self.C_Z = np.asarray(C_Z_prices, dtype=float)

        self.E_X  = float(E_X)
        self.E_Y  = float(E_Y)
        self.beta = float(beta)

        self.exotic_payoff_matrix = (
            np.asarray(exotic_payoff_matrix, dtype=float)
            if exotic_payoff_matrix is not None else None
        )

        self._epsilon = float(epsilon)

        self._n = len(x_nodes)
        self._m = len(y_nodes)

        self._nK_X = len(K_X)
        self._nK_Y = len(K_Y)
        self._nK_Z = len(K_Z)
        self._n_params = 3 + self._nK_X + self._nK_Y + self._nK_Z

        self._build_basis_matrices()
        self._build_linear_coefficients()

    # ------------------------------------------------------------------
    # epsilon property (allows annealing from outside)
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Entropic regularisation parameter."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"epsilon must be positive, got {value}.")
        self._epsilon = float(value)

    # ------------------------------------------------------------------
    # Basis construction
    # ------------------------------------------------------------------

    def _build_basis_matrices(self) -> None:
        """Pre-compute the stacked basis tensor Phi of shape (n_params, n, m)."""
        n, m = self._n, self._m
        x = self.x_nodes    # (n,)
        y = self.y_nodes    # (m,)

        # Broadcast helpers
        X = x[:, np.newaxis]                    # (n, 1)
        Y = y[np.newaxis, :]                    # (1, m)
        Z = X - Y                               # (n, m)  spread

        layers = []

        # Constant, linear-X, linear-Y
        layers.append(np.ones((n, m)))           # phi_norm
        layers.append(np.broadcast_to(X, (n, m)).copy())  # phi_x
        layers.append(np.broadcast_to(Y, (n, m)).copy())  # phi_y

        # X call payoffs
        for r in self.K_X:
            layers.append(_relu(X - r) * np.ones((n, m)))

        # Y call payoffs
        for r in self.K_Y:
            layers.append(_relu(Y - r) * np.ones((n, m)))

        # Z = X-Y call payoffs (spread options)
        for r in self.K_Z:
            layers.append(_relu(Z - r))

        # Stack into (n_params, n, m)
        self._Phi = np.stack(layers, axis=0)   # (n_params, n, m)
        # Flattened view for Hessian computation
        self._Phi_flat = self._Phi.reshape(self._n_params, n * m)

    def _build_linear_coefficients(self) -> None:
        """Build the vector b so that linear_part = theta @ b."""
        # b = [1, E_X, E_Y, C_X[0..nK_X-1], C_Y[0..nK_Y-1], C_Z[0..nK_Z-1]]
        self._b = np.concatenate([
            [1.0, self.E_X, self.E_Y],
            self.C_X,
            self.C_Y,
            self.C_Z,
        ])

    # ------------------------------------------------------------------
    # Parameter unpacking
    # ------------------------------------------------------------------

    def _unpack_theta(self, theta: np.ndarray
                      ) -> Tuple[float, float, float,
                                 np.ndarray, np.ndarray, np.ndarray]:
        """Split theta into (m0, m1, m2, q_X, q_Y, q_Z)."""
        m0 = theta[0]
        m1 = theta[1]
        m2 = theta[2]
        i0 = 3
        i1 = i0 + self._nK_X
        i2 = i1 + self._nK_Y
        q_X = theta[i0:i1]
        q_Y = theta[i1:i2]
        q_Z = theta[i2:]
        return float(m0), float(m1), float(m2), q_X, q_Y, q_Z

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def _compute_A_ij(self, theta: np.ndarray) -> np.ndarray:
        """Compute the dual potential matrix A_ij of shape (n, m).

        A_ij = theta · Phi_ij  −  beta * Exotic_ij
        """
        A = np.einsum("k,kij->ij", theta, self._Phi)
        if self.beta != 0.0 and self.exotic_payoff_matrix is not None:
            A -= self.beta * self.exotic_payoff_matrix
        return A

    def _compute_W_tilde(self, theta: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute W̃_ij = Q_ij * exp((A_ij − A_max)/ε) and A_max/ε.

        Returns
        -------
        W_tilde : ndarray (n, m)
            Un-scaled weights satisfying  W_ij = W̃_ij * exp(B_max).
        B_max : float
            The shift  max(A_ij/ε) used for numerical stabilisation.
        """
        A   = self._compute_A_ij(theta)
        B   = A / self._epsilon
        B_max = float(B.max())
        W_tilde = self._Q_ij * np.exp(B - B_max)
        return W_tilde, B_max

    def _compute_W_ij(self, theta: np.ndarray) -> np.ndarray:
        """Compute W_ij = Q_ij * exp(A_ij / ε) with overflow-safe scaling.

        When exp(B_max) would overflow float64 the returned matrix is
        proportional to W_ij (scaled by exp(min(B_max, 500))). Callers that
        need the absolute scale (e.g. normalization in ``solve``) should
        divide by ``W.sum()`` themselves.
        """
        W_tilde, B_max = self._compute_W_tilde(theta)
        # Cap the exponent to float64-safe range; keeps W finite while
        # preserving the correct shape and direction of the distribution.
        scale = np.exp(min(float(B_max), 500.0))
        return W_tilde * scale

    # ------------------------------------------------------------------
    # Objective and gradient
    # ------------------------------------------------------------------

    def objective_and_gradient(self, theta: np.ndarray
                                ) -> Tuple[float, np.ndarray]:
        """Compute the dual objective L(theta) and its gradient.

        Parameters
        ----------
        theta : ndarray, shape (n_params,)
            Current dual variable vector.

        Returns
        -------
        L : float
            Dual objective value (to be maximised).
        grad : ndarray, shape (n_params,)
            Gradient ∂L/∂theta.
        """
        theta = np.asarray(theta, dtype=float)

        W_tilde, B_max = self._compute_W_tilde(theta)

        sum_W_tilde = W_tilde.sum()
        linear_part = float(theta @ self._b)

        # Work through log(Z) = B_max + log(sum_W_tilde) to avoid overflow.
        log_Z = B_max + np.log(max(sum_W_tilde, 1e-300))

        # Normalized weight matrix: always finite regardless of B_max.
        W_norm = W_tilde / max(sum_W_tilde, 1e-300)
        # E_{W_norm}[Phi_k]: finite even when exp(B_max) overflows.
        phi_expectation = np.einsum("kij,ij->k", self._Phi, W_norm)

        if log_Z > 700.0:
            # exp(log_Z) would overflow → objective is effectively −∞.
            # Return a large negative L so the line-search backtracks,
            # and a gradient that indicates the direction to decrease Z.
            L = -1e300
            grad = self._b - 1e300 * phi_expectation
        else:
            Z = np.exp(log_Z)
            L = linear_part - self._epsilon * Z + self._epsilon
            # Gradient: ∂L/∂theta_k = b_k − Z * E_{W_norm}[Phi_k]
            grad = self._b - Z * phi_expectation

        return L, grad

    # ------------------------------------------------------------------
    # Hessian
    # ------------------------------------------------------------------

    def _normalized_newton_data(self, theta: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return the normalized Hessian, the normalized RHS, and log_Z.

        The true Newton system is  H * delta = −g  where
            H   = −(Z/ε) M_norm   (Z may overflow)
            g   = b − Z φ_avg
        Dividing by Z gives the equivalent finite system:
            H_norm * delta = rhs   where  rhs = φ_avg − b/Z = −g/Z
        which is used in the solver.  H_norm = −(1/ε) M_norm.

        Returns
        -------
        H_norm : ndarray (n_params, n_params)
        rhs    : ndarray (n_params,)   i.e. φ_avg − b/Z
        log_Z  : float
        """
        W_tilde, B_max = self._compute_W_tilde(theta)
        sum_W  = max(float(W_tilde.sum()), 1e-300)
        log_Z  = float(B_max) + np.log(sum_W)
        W_norm = W_tilde / sum_W                       # always finite

        phi_avg = np.einsum("kij,ij->k", self._Phi, W_norm)  # finite

        if log_Z > 700.0:
            b_over_Z = np.zeros(self._n_params)
        else:
            b_over_Z = self._b / np.exp(log_Z)

        rhs    = phi_avg - b_over_Z                           # −g/Z, finite
        W_flat = W_norm.flatten()
        H_norm = (-(1.0 / self._epsilon)
                  * (self._Phi_flat * W_flat) @ self._Phi_flat.T)
        return H_norm, rhs, log_Z

    def hessian(self, theta: np.ndarray,
                W: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the Hessian H of the dual objective L(theta).

        H_kl = −(1/ε) Σ_ij W_ij Φ_k_ij Φ_l_ij = −(Z/ε) M_norm_kl

        The Hessian is symmetric negative semi-definite.  For reporting and
        tests the method accepts a pre-computed W (used directly).  Internally
        the solver uses ``_normalized_newton_data`` to avoid overflow.

        Parameters
        ----------
        theta : ndarray, shape (n_params,)
        W : ndarray (n, m) or None
            Pre-computed W_ij.  If None the computation goes through the
            overflow-safe normalized path.

        Returns
        -------
        H : ndarray, shape (n_params, n_params)
        """
        if W is not None:
            # Caller supplied W — use it directly (external use / tests).
            W_flat = W.flatten()
            return -(1.0 / self._epsilon) * (self._Phi_flat * W_flat) @ self._Phi_flat.T

        # Overflow-safe path: H = Z * H_norm
        H_norm, _, log_Z = self._normalized_newton_data(theta)
        if log_Z > 700.0:
            return 1e300 * H_norm          # large but finite matrix
        return np.exp(log_Z) * H_norm

    # ------------------------------------------------------------------
    # Newton step
    # ------------------------------------------------------------------

    def newton_step(self, theta: np.ndarray,
                    epsilon_current: Optional[float] = None
                    ) -> np.ndarray:
        """Compute one Newton direction  delta = H^{-1}(−g).

        Uses the normalized system  H_norm * delta = rhs  (where rhs = −g/Z)
        to avoid float64 overflow when Z = Σ W_ij is very large.

        Parameters
        ----------
        theta : ndarray
            Current iterate.
        epsilon_current : float or None
            Override for epsilon (used during annealing).

        Returns
        -------
        delta : ndarray, shape (n_params,)
            Ascent Newton direction.
        """
        if epsilon_current is not None:
            old_eps = self._epsilon
            self._epsilon = epsilon_current

        _, grad = self.objective_and_gradient(theta)
        H_norm, rhs, _ = self._normalized_newton_data(theta)

        if epsilon_current is not None:
            self._epsilon = old_eps

        # Solve  H_norm @ delta = rhs  (equivalent to H delta = −g but finite).
        # H_norm is symmetric negative-semidefinite; use eigendecomposition so
        # near-zero eigenvalues are filtered rather than inverted.
        eigvals, eigvecs = np.linalg.eigh(H_norm)
        threshold = max(float(np.abs(eigvals).max()) * 1e-8, 1e-14)
        mask = eigvals < -threshold          # keep only meaningful negatives
        if not np.any(mask):
            # H_norm ≈ 0 → fall back to (scaled) gradient direction.
            norm_g = np.linalg.norm(grad)
            return grad / (norm_g + 1e-15)
        inv_eigvals = np.where(mask, 1.0 / eigvals, 0.0)
        delta = eigvecs @ (inv_eigvals * (eigvecs.T @ rhs))
        return delta

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def solve(self,
              theta_init: Optional[np.ndarray] = None,
              max_iter:   int   = 500,
              tol:        float = 1e-6,
              verbose:    bool  = False) -> Dict:
        """Damped-Newton solver for the dual objective.

        Uses an Armijo backtracking line-search to guarantee ascent.

        Parameters
        ----------
        theta_init : ndarray or None
            Initial dual variable vector (zeros if None).
        max_iter : int
            Maximum number of Newton iterations (default 100).
        tol : float
            Convergence tolerance on  max |grad| (default 1e-8).
        verbose : bool
            Print iteration progress when True.

        Returns
        -------
        dict with keys
            ``theta``            : ndarray — optimal dual variables
            ``converged``        : bool
            ``iterations``       : int
            ``final_grad_norm``  : float  — max absolute gradient
            ``W_ij``             : ndarray (n,m) — recovered joint weights
            ``L``                : float — optimal objective value
        """
        if theta_init is None:
            theta = np.zeros(self._n_params)
        else:
            theta = np.asarray(theta_init, dtype=float).copy()

        converged   = False
        L_val       = -np.inf
        grad        = np.zeros_like(theta)
        armijo_c    = 0.01

        for it in range(max_iter):
            L_val, grad = self.objective_and_gradient(theta)
            grad_norm = float(np.max(np.abs(grad)))

            if verbose:
                print(f"  iter {it:4d}  L = {L_val:+.8f}  |g|_inf = {grad_norm:.3e}")

            if grad_norm < tol:
                converged = True
                break

            # Newton direction — use normalized system H_norm * delta = rhs
            # to avoid float64 overflow when Z = Σ W_ij is very large.
            H_norm, rhs, _ = self._normalized_newton_data(theta)

            # Adaptive diagonal regularisation: scale by the RMS diagonal entry.
            diag_abs = np.abs(np.diag(H_norm))
            diag_rms = float(np.sqrt(np.mean(diag_abs ** 2))) if len(diag_abs) else 1.0
            diag_scale = max(1e-14, 1e-8 * diag_rms)
            H_reg = H_norm - diag_scale * np.eye(self._n_params)

            try:
                delta = np.linalg.solve(H_reg, rhs)
            except np.linalg.LinAlgError:
                lam = max(1e-10, 1e-6 * diag_rms)
                H_lm = H_norm - lam * np.eye(self._n_params)
                delta, _, _, _ = np.linalg.lstsq(H_lm, rhs, rcond=None)

            slope = float(grad @ delta)
            if not np.isfinite(slope) or slope <= 0.0:
                # Newton direction failed.  rhs = φ_avg − b/Z ≈ +φ_avg when
                # Z is large, but the ascent direction is −φ_avg (reduces Z).
                # Use −rhs as the fallback steepest-descent direction.
                rhs_norm = float(np.linalg.norm(rhs))
                if rhs_norm > 0.0:
                    delta = -rhs / rhs_norm   # ascent: decreases Z
                    slope = float(grad @ delta)
                if not np.isfinite(slope) or slope <= 0.0:
                    warnings.warn("Fallback direction also non-ascent; stopping.")
                    break

            # Armijo backtracking line-search
            alpha = 1.0
            for _ in range(30):
                L_new, _ = self.objective_and_gradient(theta + alpha * delta)
                if L_new > L_val + armijo_c * alpha * slope:
                    break
                alpha *= 0.5

            theta = theta + alpha * delta

        # Final evaluation
        L_val, grad = self.objective_and_gradient(theta)
        grad_norm = float(np.max(np.abs(grad)))
        if grad_norm < tol:
            converged = True

        W_tilde, B_max = self._compute_W_tilde(theta)
        W_ij = W_tilde / max(float(W_tilde.sum()), 1e-300)   # normalised, always finite

        return {
            "theta":           theta,
            "converged":       converged,
            "iterations":      it + 1,
            "final_grad_norm": grad_norm,
            "W_ij":            W_ij,
            "L":               float(L_val),
        }

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def recover_joint_weights(self, theta: np.ndarray) -> np.ndarray:
        """Return W_ij (normalised) for the given dual variable *theta*.

        Parameters
        ----------
        theta : ndarray

        Returns
        -------
        W_ij : ndarray, shape (n, m), sums to 1
        """
        W_tilde, _ = self._compute_W_tilde(theta)
        W_tilde /= max(float(W_tilde.sum()), 1e-300)
        return W_tilde

    def compute_exotic_price(self, W_ij: np.ndarray,
                             payoff_matrix: np.ndarray) -> float:
        """Price an exotic option as the expectation under W_ij.

        Parameters
        ----------
        W_ij : ndarray (n, m)
            Joint probability weights (should sum to 1).
        payoff_matrix : ndarray (n, m)
            Payoff at each node pair (x_i, y_j).

        Returns
        -------
        float
        """
        return float(np.sum(W_ij * payoff_matrix))

    def model_call_prices(self, W_ij: np.ndarray,
                          strikes: np.ndarray,
                          asset: str = "X") -> np.ndarray:
        """Compute model call prices from the joint distribution W_ij.

        Parameters
        ----------
        W_ij : ndarray (n, m)
        strikes : ndarray
            Strike vector for the desired asset.
        asset : {'X', 'Y', 'Z'}
            Which call options to price.
            'Z' prices  max(X − Y − K, 0)  (spread call).

        Returns
        -------
        prices : ndarray, same shape as *strikes*
        """
        asset = asset.upper()
        x = self.x_nodes    # (n,)
        y = self.y_nodes    # (m,)

        prices = np.empty(len(strikes))
        for i, K in enumerate(strikes):
            if asset == "X":
                payoff = _relu(x[:, np.newaxis] - K) * np.ones((self._n, self._m))
            elif asset == "Y":
                payoff = _relu(y[np.newaxis, :] - K) * np.ones((self._n, self._m))
            elif asset == "Z":
                payoff = _relu(x[:, np.newaxis] - y[np.newaxis, :] - K)
            else:
                raise ValueError(f"asset must be 'X', 'Y', or 'Z', got '{asset}'.")
            prices[i] = float(np.sum(W_ij * payoff))
        return prices


# ---------------------------------------------------------------------------
# Epsilon annealing schedule
# ---------------------------------------------------------------------------

def adaptive_epsilon_schedule(epsilon_target: float,
                               epsilon_start:  Optional[float] = None
                               ) -> np.ndarray:
    """Geometric annealing schedule from epsilon_start down to epsilon_target.

    The schedule follows the paper's recommended:
        ε_k = ε_start * (ε_target / ε_start)^((k-1)/(K-1)),  k = 1 … K

    so ε_1 = ε_start and ε_K = ε_target.

    Parameters
    ----------
    epsilon_target : float
        Final (smallest) epsilon value.
    epsilon_start : float or None
        Initial epsilon. Defaults to  max(0.05, 10 * epsilon_target).

    Returns
    -------
    schedule : ndarray, shape (K,)
        Geometrically decreasing epsilon values.
    """
    if epsilon_start is None:
        epsilon_start = max(0.05, 10.0 * epsilon_target)

    if epsilon_target <= 0:
        raise ValueError(f"epsilon_target must be positive, got {epsilon_target}.")
    if epsilon_start <= epsilon_target:
        raise ValueError(
            f"epsilon_start ({epsilon_start}) must exceed "
            f"epsilon_target ({epsilon_target})."
        )

    ratio    = epsilon_target / epsilon_start
    K_steps  = max(5, int(np.floor(-2.0 * np.log(ratio))))

    t        = np.linspace(0.0, 1.0, K_steps)   # 0 → 1
    schedule = epsilon_start * (ratio ** t)

    return schedule
