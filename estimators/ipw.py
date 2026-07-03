"""
Inverse Probability Weighting Estimator
=======================================

Implements ATT-IPW as described in Section 4.3 of the paper.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..config import SimulationConfig
from ..utils import clip_weights
from .base import BaseEstimator, EstimatorResult, normalize_weights


class IPWEstimator(BaseEstimator):
    """
    ATT Inverse Probability Weighting estimator.
    
    Treated units: weight = 1/n1
    Control units: odds weight = ehat / (1 - ehat), normalized to sum to 1
    
    Target: ATT
    """
    
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
    
    def estimate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        ehat: np.ndarray,
        tau_x: Optional[np.ndarray] = None,
        **kwargs
    ) -> EstimatorResult:
        """
        Estimate ATT using IPW.
        
        Parameters
        ----------
        X : np.ndarray
            Covariate matrix (n, 6)
        T : np.ndarray
            Treatment indicator
        Y : np.ndarray
            Observed outcome
        ehat : np.ndarray
            Estimated propensity scores
        tau_x : np.ndarray, optional
            True individual treatment effects (for estimand drift)
        
        Returns
        -------
        EstimatorResult
            IPW estimation result
        """
        n = len(Y)
        
        treated = T == 1
        control = T == 0
        
        n1 = int(np.sum(treated))
        n0 = int(np.sum(control))
        
        if n1 == 0 or n0 == 0:
            return EstimatorResult.invalid("IPW", n, "No treated or control units")
        
        # Control odds weights
        odds = ehat[control] / (1.0 - ehat[control])
        denom = np.sum(odds)
        
        if denom <= 0:
            return EstimatorResult.invalid("IPW", n, "Odds sum to zero")
        
        # Construct weights
        treated_weights = np.zeros(n)
        control_weights = np.zeros(n)
        
        treated_weights[treated] = 1.0 / n1
        control_weights[control] = odds / denom
        
        # Clip weights
        treated_weights = clip_weights(treated_weights)
        control_weights = clip_weights(control_weights)
        
        # Compute estimate
        tau_hat = np.sum(treated_weights * Y) - np.sum(control_weights * Y)
        
        # IPW targets ATT directly (no estimand drift)
        estimand_drift = 0.0
        
        # Compute confidence interval
        lower, upper = self._approximate_ci(
            tau_hat, Y, treated_weights, control_weights, self.config.alpha
        )
        
        # Compute weight statistics
        ess_control = self._effective_sample_size(control_weights[control])
        max_weight = np.max(control_weights[control]) if n0 > 0 else np.nan
        
        return EstimatorResult(
            method="IPW",
            tau_hat=float(tau_hat),
            treated_weights=treated_weights,
            control_weights=control_weights,
            estimand_drift=estimand_drift,
            valid=True,
            metadata={
                "lower": lower,
                "upper": upper,
                "ess_control": ess_control,
                "max_weight": max_weight,
                "n_treated": n1,
                "n_control": n0,
            }
        )
    
    @staticmethod
    def _effective_sample_size(weights: np.ndarray) -> float:
        """Compute Kish effective sample size."""
        if np.sum(weights) <= 0:
            return np.nan
        return float(np.sum(weights) ** 2 / np.sum(weights ** 2))