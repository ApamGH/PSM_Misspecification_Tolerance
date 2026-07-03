"""
Augmented Inverse Probability Weighting Estimator
================================================

Implements AIPW as described in Section 4.6 of the paper.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..config import SimulationConfig
from ..utils import clip_weights
from .base import BaseEstimator, EstimatorResult


class AIPWEstimator(BaseEstimator):
    """
    ATT-oriented Augmented Inverse Probability Weighting estimator.
    
    Combines propensity-score weighting with outcome-regression adjustment.
    Available in two versions:
    - AIPW_correct: uses extended outcome model (correct when nonlinear)
    - AIPW_restricted: uses restricted linear outcome model
    
    Target: ATT
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        method_name: str = "AIPW",
        use_extended_outcome: bool = False
    ):
        super().__init__(config)
        self.method_name = method_name
        self.use_extended_outcome = use_extended_outcome
    
    def estimate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        ehat: np.ndarray,
        m0hat: Optional[np.ndarray] = None,
        tau_x: Optional[np.ndarray] = None,
        **kwargs
    ) -> EstimatorResult:
        """
        Estimate ATT using AIPW.
        
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
        m0hat : np.ndarray, optional
            Estimated outcome regression (if None, will be estimated)
        tau_x : np.ndarray, optional
            True individual treatment effects (for estimand drift)
        
        Returns
        -------
        EstimatorResult
            AIPW estimation result
        """
        n = len(Y)
        
        treated = T == 1
        control = T == 0
        
        n1 = int(np.sum(treated))
        n0 = int(np.sum(control))
        
        if n1 == 0 or n0 == 0:
            return EstimatorResult.invalid(self.method_name, n, "No treated or control units")
        
        # If m0hat not provided, estimate it
        if m0hat is None:
            from ..data_generation import fit_outcome_control
            m0hat = fit_outcome_control(X, T, Y, extended=self.use_extended_outcome)
        
        # Control odds weights
        odds = ehat[control] / (1.0 - ehat[control])
        denom = np.sum(odds)
        
        if denom <= 0:
            return EstimatorResult.invalid(self.method_name, n, "Odds sum to zero")
        
        # AIPW estimator (ATT-oriented)
        # tau_hat = mean_treated(Y - m0hat) - weighted_mean_control(Y - m0hat)
        treated_component = np.mean(Y[treated] - m0hat[treated])
        control_component = np.sum(odds * (Y[control] - m0hat[control])) / denom
        
        tau_hat = treated_component - control_component
        
        # Construct weights (for diagnostics)
        treated_weights = np.zeros(n)
        control_weights = np.zeros(n)
        
        treated_weights[treated] = 1.0 / n1
        control_weights[control] = odds / denom
        
        # Clip weights
        treated_weights = clip_weights(treated_weights)
        control_weights = clip_weights(control_weights)
        
        # AIPW targets ATT directly
        estimand_drift = 0.0
        
        # Compute confidence interval
        lower, upper = self._approximate_ci(
            tau_hat, Y, treated_weights, control_weights, self.config.alpha
        )
        
        # Compute weight statistics
        ess_control = self._effective_sample_size(control_weights[control])
        max_weight = np.max(control_weights[control]) if n0 > 0 else np.nan
        
        return EstimatorResult(
            method=self.method_name,
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
                "use_extended_outcome": self.use_extended_outcome,
            }
        )
    
    @staticmethod
    def _effective_sample_size(weights: np.ndarray) -> float:
        """Compute Kish effective sample size."""
        if np.sum(weights) <= 0:
            return np.nan
        return float(np.sum(weights) ** 2 / np.sum(weights ** 2))