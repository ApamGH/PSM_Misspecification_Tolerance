"""
Entropy Balancing Estimator
===========================

Implements ATT entropy balancing using the dual optimisation problem.
The dual form is much faster and more stable than solving one equality
constraint per balance moment with SLSQP, which is important for local-machine
simulation runs.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from scipy.optimize import minimize
from scipy.special import logsumexp

from ..config import SimulationConfig
from ..utils import clip_weights, standardize_basis
from .base import BaseEstimator, EstimatorResult


class EBEstimator(BaseEstimator):
    """
    Entropy Balancing estimator for the ATT.

    Control weights are chosen so that weighted control moments approximate
    treated moments for the selected balance basis. The implementation solves
    the dual entropy-balancing problem and then recovers primal weights.
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.extended_balance = config.eb_extended_balance
        self.maxiter = config.eb_maxiter
        self.tolerance = config.eb_tolerance

    def estimate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        ehat: np.ndarray,
        tau_x: Optional[np.ndarray] = None,
        **kwargs,
    ) -> EstimatorResult:
        n = len(Y)
        treated = T == 1
        control = T == 0
        n1 = int(np.sum(treated))
        n0 = int(np.sum(control))

        if n1 == 0 or n0 == 0:
            return EstimatorResult.invalid("EB", n, "No treated or control units")

        H = standardize_basis(self._balance_basis(X))
        H_t = H[treated]
        H_c = H[control]
        target = np.mean(H_t, axis=0)
        q = H_c.shape[1]

        if n0 <= q + 1:
            return EstimatorResult.invalid("EB", n, "Too few controls for balance constraints")

        # Center controls at target moments. The dual problem seeks weights
        # with weighted mean of Z equal to zero.
        Z = H_c - target

        def objective(lam: np.ndarray) -> float:
            eta = Z @ lam
            return float(logsumexp(eta))

        def gradient(lam: np.ndarray) -> np.ndarray:
            eta = Z @ lam
            w = np.exp(eta - logsumexp(eta))
            return Z.T @ w

        result = minimize(
            fun=objective,
            x0=np.zeros(q),
            jac=gradient,
            method="BFGS",
            options={"maxiter": self.maxiter, "gtol": self.tolerance, "disp": False},
        )

        # BFGS may stop with precision-loss warnings even when balance is
        # already adequate. Accept the solution if the balance gradient is small.
        grad_norm = float(np.linalg.norm(gradient(result.x), ord=np.inf)) if result.x is not None else np.inf
        if (not result.success) and grad_norm > 1e-5:
            return EstimatorResult.invalid("EB", n, f"Dual optimisation failed: {result.message}")

        eta = Z @ result.x
        w_control = np.exp(eta - logsumexp(eta))

        if np.any(~np.isfinite(w_control)) or np.sum(w_control) <= 0:
            return EstimatorResult.invalid("EB", n, "Invalid entropy-balancing weights")

        treated_weights = np.zeros(n)
        control_weights = np.zeros(n)
        treated_weights[treated] = 1.0 / n1
        control_weights[control] = w_control / np.sum(w_control)

        treated_weights = clip_weights(treated_weights)
        control_weights = clip_weights(control_weights)

        tau_hat = np.sum(treated_weights * Y) - np.sum(control_weights * Y)
        lower, upper = self._approximate_ci(
            tau_hat, Y, treated_weights, control_weights, self.config.alpha
        )

        balance_achieved = self._compute_balance(H_t, H_c, control_weights[control])

        return EstimatorResult(
            method="EB",
            tau_hat=float(tau_hat),
            treated_weights=treated_weights,
            control_weights=control_weights,
            estimand_drift=0.0,
            valid=True,
            metadata={
                "lower": lower,
                "upper": upper,
                "balance_achieved": balance_achieved,
                "n_constraints": q,
                "n_treated": n1,
                "n_control": n0,
                "converged": bool(result.success or grad_norm <= 1e-5),
                "iterations": getattr(result, "nit", np.nan),
                "grad_norm": grad_norm,
            },
        )

    def _balance_basis(self, X: np.ndarray) -> np.ndarray:
        """
        Create the balance basis for entropy balancing.
        """
        if not self.extended_balance:
            return X

        x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
        return np.column_stack([
            x1, x2, x3, x4, x5, x6,
            x1**2, x2**2, x3**2, x4**2,
            x1 * x2, x3 * x4,
        ])

    @staticmethod
    def _compute_balance(H_t: np.ndarray, H_c: np.ndarray, w_control: np.ndarray) -> float:
        treated_mean = np.mean(H_t, axis=0)
        control_mean = np.average(H_c, axis=0, weights=w_control)
        return float(np.max(np.abs(treated_mean - control_mean)))
