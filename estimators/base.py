"""
Base Estimator Classes
======================

Defines the base classes and interfaces for all estimators.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..config import SimulationConfig
from ..utils import weighted_mean, weighted_variance, effective_sample_size


@dataclass
class EstimatorResult:
    """
    Result from an estimator.
    
    Attributes
    ----------
    method : str
        Method name
    tau_hat : float
        Estimated treatment effect
    treated_weights : np.ndarray
        Weights for treated units (length n)
    control_weights : np.ndarray
        Weights for control units (length n)
    estimand_drift : float
        Estimand drift (for methods that change treated-side representation)
    valid : bool
        Whether the estimate is valid
    metadata : Dict[str, Any]
        Additional metadata
    """
    method: str
    tau_hat: float
    treated_weights: np.ndarray
    control_weights: np.ndarray
    estimand_drift: float = 0.0
    valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def invalid(cls, method: str, n: int, reason: str = "") -> "EstimatorResult":
        """Create an invalid result."""
        return cls(
            method=method,
            tau_hat=np.nan,
            treated_weights=np.zeros(n),
            control_weights=np.zeros(n),
            estimand_drift=np.nan,
            valid=False,
            metadata={"reason": reason},
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "tau_hat": self.tau_hat,
            "treated_weights": self.treated_weights,
            "control_weights": self.control_weights,
            "estimand_drift": self.estimand_drift,
            "valid": self.valid,
            **self.metadata,
        }


class BaseEstimator(ABC):
    """
    Base class for all estimators.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    @abstractmethod
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
        Estimate the treatment effect.
        
        Parameters
        ----------
        X : np.ndarray
            Covariate matrix (n, 6)
        T : np.ndarray
            Treatment indicator (0/1)
        Y : np.ndarray
            Observed outcome
        ehat : np.ndarray
            Estimated propensity scores
        tau_x : np.ndarray, optional
            True individual treatment effects (for estimand drift)
        **kwargs
            Additional method-specific arguments
        
        Returns
        -------
        EstimatorResult
            Estimation result
        """
        pass
    
    def _compute_estimand_drift(
        self,
        treated_weights: np.ndarray,
        tau_x: np.ndarray,
        T: np.ndarray
    ) -> float:
        """
        Compute estimand drift for methods that change treated-side representation.
        
        S_m(delta) = tau_m^(1)(delta) - tau_ATT(delta)
        
        Parameters
        ----------
        treated_weights : np.ndarray
            Treated weights
        tau_x : np.ndarray
            Individual treatment effects
        T : np.ndarray
            Treatment indicator
        
        Returns
        -------
        float
            Estimand drift
        """
        if tau_x is None or np.sum(treated_weights) <= 0:
            return np.nan
        
        # Method-induced treated estimand
        tau_method = np.sum(treated_weights * tau_x)
        
        # Original ATT
        tau_att = np.mean(tau_x[T == 1])
        
        return float(tau_method - tau_att)
    
    def _approximate_ci(
        self,
        tau_hat: float,
        Y: np.ndarray,
        treated_weights: np.ndarray,
        control_weights: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Approximate confidence interval using weighted variance components.
        
        For final article-level simulation, bootstrap should be used instead.
        This is kept as a fallback.
        
        Parameters
        ----------
        tau_hat : float
            Point estimate
        Y : np.ndarray
            Observed outcome
        treated_weights : np.ndarray
            Treated weights
        control_weights : np.ndarray
            Control weights
        alpha : float
            Significance level
        
        Returns
        -------
        Tuple[float, float]
            (lower, upper) confidence limits
        """
        z = 1.959963984540054  # 97.5% quantile of normal
        
        treated_used = treated_weights > 0
        control_used = control_weights > 0
        
        if np.sum(treated_used) < 2 or np.sum(control_used) < 2:
            return (np.nan, np.nan)
        
        ess_t = effective_sample_size(treated_weights[treated_used])
        ess_c = effective_sample_size(control_weights[control_used])
        
        var_t = weighted_variance(Y[treated_used], treated_weights[treated_used])
        var_c = weighted_variance(Y[control_used], control_weights[control_used])
        
        if np.isnan(ess_t) or np.isnan(ess_c) or ess_t <= 1 or ess_c <= 1:
            return (np.nan, np.nan)
        
        se = np.sqrt(var_t / ess_t + var_c / ess_c)
        
        if not np.isfinite(se):
            return (np.nan, np.nan)
        
        return (
            float(tau_hat - z * se),
            float(tau_hat + z * se),
        )


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights to sum to 1.
    
    Parameters
    ----------
    weights : np.ndarray
        Weights to normalize
    
    Returns
    -------
    np.ndarray
        Normalized weights
    """
    weight_sum = np.sum(weights)
    if weight_sum <= 0:
        return np.zeros_like(weights)
    return weights / weight_sum


def clip_weights(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Clip only positive weights to avoid numerical issues.

    Structural zero weights must remain zero because they identify units
    that are not part of a method's effective analysis support. Turning
    zeros into small positive weights would incorrectly make excluded
    units appear to be retained in effective-support diagnostics.
    """
    weights = np.asarray(weights, dtype=float).copy()
    positive = weights > 0
    weights[positive] = np.clip(weights[positive], eps, None)
    return weights