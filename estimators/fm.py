"""
Full Matching Estimator
=======================

Implements Full Matching (approximation) as described in Section 4.2 of the paper.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List

from ..config import SimulationConfig
from ..utils import clip_weights
from .base import BaseEstimator, EstimatorResult


class FMEstimator(BaseEstimator):
    """
    Full Matching estimator (approximation using subclasses).
    
    Approximates full matching by creating propensity-score subclasses
    where each subclass contains at least one treated and one control unit.
    
    Target: ATT for retained treated units.
    """
    
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.initial_subclasses = config.fm_initial_subclasses
        self.min_subclass_size = config.fm_min_subclass_size
    
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
        Estimate ATT using Full Matching approximation.
        
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
            FM estimation result
        """
        n = len(Y)
        
        # Create subclasses
        subclasses = self._make_subclasses(ehat, T)
        
        valid = subclasses >= 0
        
        if not np.any(valid):
            return EstimatorResult.invalid("FM", n, "No valid subclasses")
        
        # Get valid subclasses
        valid_subclasses = np.unique(subclasses[valid])
        
        # Count retained treated units
        n1_total = int(np.sum((T == 1) & valid))
        
        if n1_total <= 0:
            return EstimatorResult.invalid("FM", n, "No retained treated units")
        
        # Construct weights
        treated_weights = np.zeros(n)
        control_weights = np.zeros(n)
        
        for g in valid_subclasses:
            idx_g = subclasses == g
            
            treated_g = idx_g & (T == 1)
            control_g = idx_g & (T == 0)
            
            n1_g = int(np.sum(treated_g))
            n0_g = int(np.sum(control_g))
            
            if n1_g <= 0 or n0_g <= 0:
                continue
            
            # ATT-oriented subclass weighting
            # Treated: 1/n1_total per retained treated unit
            # Control: n1_g / (n1_total * n0_g) per control unit
            treated_weights[treated_g] = 1.0 / n1_total
            control_weights[control_g] = n1_g / (n1_total * n0_g)
        
        # Clip weights
        treated_weights = clip_weights(treated_weights)
        control_weights = clip_weights(control_weights)
        
        if np.sum(treated_weights) <= 0 or np.sum(control_weights) <= 0:
            return EstimatorResult.invalid("FM", n, "Weight sums to zero")
        
        # Compute estimate
        tau_hat = np.sum(treated_weights * Y) - np.sum(control_weights * Y)
        
        # Compute estimand drift
        estimand_drift = self._compute_estimand_drift(treated_weights, tau_x, T)
        
        # Compute confidence interval
        lower, upper = self._approximate_ci(
            tau_hat, Y, treated_weights, control_weights, self.config.alpha
        )
        
        return EstimatorResult(
            method="FM",
            tau_hat=float(tau_hat),
            treated_weights=treated_weights,
            control_weights=control_weights,
            estimand_drift=estimand_drift,
            valid=True,
            metadata={
                "lower": lower,
                "upper": upper,
                "n_subclasses": len(valid_subclasses),
                "n_retained_treated": n1_total,
                "n_treated_original": int(np.sum(T == 1)),
                "retention_rate": n1_total / int(np.sum(T == 1)),
            }
        )
    
    def _make_subclasses(
        self,
        ehat: np.ndarray,
        T: np.ndarray
    ) -> np.ndarray:
        """
        Create subclasses for full matching approximation.
        
        Forms initial quantile-based subclasses and merges adjacent
        subclasses to ensure each subclass has both treated and control units.
        """
        n = len(ehat)
        
        order = np.argsort(ehat)
        subclasses = np.full(n, -1, dtype=int)
        
        # Initial subclass assignment by quantiles
        G = min(self.initial_subclasses, n)
        ordered_groups = np.array_split(order, G)
        
        groups = [list(idx) for idx in ordered_groups if len(idx) > 0]
        
        # Merge adjacent invalid groups
        merged_groups: List[List[int]] = []
        buffer_group: List[int] = []
        
        for group in groups:
            if len(buffer_group) == 0:
                buffer_group = list(group)
            else:
                buffer_group.extend(group)
            
            t_count = int(np.sum(T[buffer_group] == 1))
            c_count = int(np.sum(T[buffer_group] == 0))
            
            if (t_count > 0 and c_count > 0 and 
                len(buffer_group) >= self.min_subclass_size):
                merged_groups.append(buffer_group)
                buffer_group = []
        
        # If remaining buffer exists, attach to last valid group
        if len(buffer_group) > 0:
            if len(merged_groups) > 0:
                merged_groups[-1].extend(buffer_group)
            else:
                merged_groups.append(buffer_group)
        
        # Final assignment
        final_subclasses = np.full(n, -1, dtype=int)
        
        for g, idx in enumerate(merged_groups):
            idx_array = np.asarray(idx, dtype=int)
            t_count = int(np.sum(T[idx_array] == 1))
            c_count = int(np.sum(T[idx_array] == 0))
            
            if t_count > 0 and c_count > 0:
                final_subclasses[idx_array] = g
        
        return final_subclasses