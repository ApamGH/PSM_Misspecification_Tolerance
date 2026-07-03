"""
Overlap Weighting Estimator
===========================

Implements OW as described in Section 4.4 of the paper.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..config import SimulationConfig
from ..utils import clip_weights
from .base import BaseEstimator, EstimatorResult, normalize_weights


class OWEstimator(BaseEstimator):
    """
    Overlap Weighting estimator.
    
    Treated units: weight = 1 - ehat
    Control units: weight = ehat
    
    Target: Average Treatment Effect in the Overlap population (ATO)
    
    Note: OW targets the overlap population, not ATT. The estimand drift
    is computed as the difference between ATO and ATT.
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
        e0: Optional[np.ndarray] = None,
        **kwargs
    ) -> EstimatorResult:
        """
        Estimate ATO using Overlap Weighting.
        
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
        e0 : np.ndarray, optional
            True propensity scores (for true ATO)
        
        Returns
        -------
        EstimatorResult
            OW estimation result
        """
        n = len(Y)
        
        treated = T == 1
        control = T == 0
        
        n1 = int(np.sum(treated))
        n0 = int(np.sum(control))
        
        if n1 == 0 or n0 == 0:
            return EstimatorResult.invalid("OW", n, "No treated or control units")
        
        # Overlap weights (unnormalized)
        w_t = 1.0 - ehat[treated]
        w_c = ehat[control]
        
        if np.sum(w_t) <= 0 or np.sum(w_c) <= 0:
            return EstimatorResult.invalid("OW", n, "Weight sums to zero")
        
        # Construct weights
        treated_weights = np.zeros(n)
        control_weights = np.zeros(n)
        
        treated_weights[treated] = w_t / np.sum(w_t)
        control_weights[control] = w_c / np.sum(w_c)
        
        # Clip weights
        treated_weights = clip_weights(treated_weights)
        control_weights = clip_weights(control_weights)
        
        # Compute ATO estimate
        tau_hat = np.sum(treated_weights * Y) - np.sum(control_weights * Y)
        
        # Compute estimand drift (ATO vs ATT)
        # Note: This requires true tau_x and e0
        estimand_drift = self._compute_estimand_drift_ow(treated_weights, tau_x, T, e0)
        
        # Compute confidence interval
        lower, upper = self._approximate_ci(
            tau_hat, Y, treated_weights, control_weights, self.config.alpha
        )
        
        # Compute weight statistics
        ess_t = self._effective_sample_size(treated_weights[treated])
        ess_c = self._effective_sample_size(control_weights[control])
        
        return EstimatorResult(
            method="OW",
            tau_hat=float(tau_hat),
            treated_weights=treated_weights,
            control_weights=control_weights,
            estimand_drift=estimand_drift,
            valid=True,
            metadata={
                "lower": lower,
                "upper": upper,
                "ess_treated": ess_t,
                "ess_control": ess_c,
                "n_treated": n1,
                "n_control": n0,
            }
        )
    
    def _compute_estimand_drift_ow(
        self,
        treated_weights: np.ndarray,
        tau_x: Optional[np.ndarray],
        T: np.ndarray,
        e0: Optional[np.ndarray]
    ) -> float:
        """
        Compute estimand drift for OW: ATO - ATT.
        
        This is not a bias but an estimand difference due to targeting
        the overlap population.
        """
        if tau_x is None or e0 is None:
            return np.nan
        
        # Compute true ATO
        weights = e0 * (1.0 - e0)
        if np.sum(weights) <= 0:
            return np.nan
        
        tau_ato = np.sum(weights * tau_x) / np.sum(weights)
        
        # Compute ATT
        tau_att = np.mean(tau_x[T == 1])
        
        return float(tau_ato - tau_att)
    
    @staticmethod
    def _effective_sample_size(weights: np.ndarray) -> float:
        """Compute Kish effective sample size."""
        if np.sum(weights) <= 0:
            return np.nan
        return float(np.sum(weights) ** 2 / np.sum(weights ** 2))