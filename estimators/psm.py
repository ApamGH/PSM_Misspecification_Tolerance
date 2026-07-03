"""
Propensity Score Matching Estimator
===================================

Implements PSM as described in Section 4.1 of the paper.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors

from ..config import SimulationConfig
from ..utils import safe_logit, clip_weights
from .base import BaseEstimator, EstimatorResult, normalize_weights


class PSMEstimator(BaseEstimator):
    """
    Propensity Score Matching estimator.
    
    One-to-one nearest-neighbour matching on the logit of the estimated
    propensity score with caliper.
    
    Target: ATT for the retained treated sample.
    """
    
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.neighbors = config.psm_neighbors
        self.caliper_multiplier = config.psm_caliper_multiplier
    
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
        Estimate ATT using PSM.
        
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
            PSM estimation result
        """
        n = len(Y)
        
        # Get treated and control indices
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]
        
        if len(treated_idx) == 0 or len(control_idx) == 0:
            return EstimatorResult.invalid("PSM", n, "No treated or control units")
        
        # Match on logit of propensity score
        logit_e = safe_logit(ehat)
        
        # Caliper: 0.20 * SD(logit(e)) as per Austin (2011b)
        caliper = self.caliper_multiplier * np.std(logit_e, ddof=1)
        
        # Prepare for matching
        treated_scores = logit_e[treated_idx].reshape(-1, 1)
        control_scores = logit_e[control_idx].reshape(-1, 1)
        
        k = min(self.neighbors, len(control_idx))
        
        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(control_scores)
        distances, indices = nn.kneighbors(treated_scores)
        
        # Apply caliper and collect matches
        retained_treated = []
        matched_controls = []
        
        for row, original_treated in enumerate(treated_idx):
            eligible = distances[row, :] <= caliper
            
            if np.any(eligible):
                selected_controls = control_idx[indices[row, eligible]]
                retained_treated.append(original_treated)
                matched_controls.append(selected_controls)
        
        n_retained = len(retained_treated)
        
        if n_retained == 0:
            return EstimatorResult.invalid("PSM", n, "No matches within caliper")
        
        # Construct weights
        treated_weights = np.zeros(n)
        control_weights = np.zeros(n)
        
        for treated_i, controls_j in zip(retained_treated, matched_controls):
            treated_weights[treated_i] = 1.0 / n_retained
            for cj in controls_j:
                control_weights[cj] += 1.0 / (n_retained * len(controls_j))
        
        # Clip and normalize
        treated_weights = clip_weights(treated_weights)
        control_weights = clip_weights(control_weights)
        
        # Compute estimate
        tau_hat = np.sum(treated_weights * Y) - np.sum(control_weights * Y)
        
        # Compute estimand drift
        estimand_drift = self._compute_estimand_drift(treated_weights, tau_x, T)
        
        # Compute confidence interval
        lower, upper = self._approximate_ci(
            tau_hat, Y, treated_weights, control_weights, self.config.alpha
        )
        
        return EstimatorResult(
            method="PSM",
            tau_hat=float(tau_hat),
            treated_weights=treated_weights,
            control_weights=control_weights,
            estimand_drift=estimand_drift,
            valid=True,
            metadata={
                "n_retained": n_retained,
                "n_treated_original": len(treated_idx),
                "retention_rate": n_retained / len(treated_idx),
                "lower": lower,
                "upper": upper,
                "caliper": caliper,
                "neighbors": k,
            }
        )