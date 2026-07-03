"""
Utility Functions
=================

Common utility functions used throughout the simulation.
"""

from __future__ import annotations

import numpy as np
import warnings
from typing import Tuple, Optional, List, Dict, Any
import logging
import pickle
from pathlib import Path


def logistic(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable logistic function.
    
    Parameters
    ----------
    z : np.ndarray
        Input array (linear predictor)
    
    Returns
    -------
    np.ndarray
        Logistic transformed values in (0, 1)
    """
    z = np.asarray(z)
    z_clipped = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Safe logit transform with clipping.
    
    Parameters
    ----------
    p : np.ndarray
        Probabilities in (0, 1)
    eps : float, optional
        Clipping threshold (default: 1e-6)
    
    Returns
    -------
    np.ndarray
        Logit transformed values
    """
    p = np.asarray(p)
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted mean with automatic weight normalisation.

    Supports both one-dimensional arrays and two-dimensional matrices.
    For matrices with shape (n, q), weights must have length n and the
    function returns a vector of q weighted column means.
    """
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)

    weight_sum = np.sum(weights)

    if weight_sum <= 0 or np.isnan(weight_sum):
        if values.ndim == 1:
            return np.nan
        return np.full(values.shape[1], np.nan)

    w = weights / weight_sum

    if values.ndim == 1:
        return float(np.sum(w * values))

    return np.sum(values * w[:, None], axis=0)


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted variance using normalised weights.
    
    Parameters
    ----------
    values : np.ndarray
        Values
    weights : np.ndarray
        Weights (will be normalised)
    
    Returns
    -------
    float
        Weighted variance
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    
    weight_sum = np.sum(weights)
    
    if weight_sum <= 0 or np.isnan(weight_sum):
        return np.nan
    
    w = weights / weight_sum
    mu = np.sum(w * values)
    
    return float(np.sum(w * (values - mu) ** 2))


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Kish effective sample size.
    
    ESS = (sum w)^2 / sum(w^2)
    
    Parameters
    ----------
    weights : np.ndarray
        Weights
    
    Returns
    -------
    float
        Effective sample size
    """
    weights = np.asarray(weights)
    
    numerator = np.sum(weights) ** 2
    denominator = np.sum(weights ** 2)
    
    if denominator <= 0 or np.isnan(denominator):
        return np.nan
    
    return float(numerator / denominator)


def standardize_basis(H: np.ndarray) -> np.ndarray:
    """
    Standardise the columns of a basis matrix.
    
    Prevents variables on larger scales from dominating the L2 norm.
    
    Parameters
    ----------
    H : np.ndarray
        Basis matrix (n_samples x n_features)
    
    Returns
    -------
    np.ndarray
        Standardised basis matrix
    """
    mean = np.mean(H, axis=0)
    sd = np.std(H, axis=0, ddof=1)
    
    # Avoid division by zero
    sd = np.where(sd <= 1e-12, 1.0, sd)
    
    return (H - mean) / sd


def setup_logger(name: str, log_file: Optional[str] = None, 
                 level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Parameters
    ----------
    name : str
        Logger name
    log_file : str, optional
        Path to log file
    level : int, optional
        Logging level (default: logging.INFO)
    
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(data: Dict[str, Any], path: str) -> None:
    """
    Save checkpoint data to file.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Data to save
    path : str
        Output path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load checkpoint data from file.
    
    Parameters
    ----------
    path : str
        Path to checkpoint file
    
    Returns
    -------
    Dict[str, Any]
        Loaded data
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


class ProgressTracker:
    """Simple progress tracker for the simulation."""
    
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self._last_percentage = -1
    
    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current_step += n
        percentage = int(100 * self.current_step / self.total_steps)
        
        if percentage > self._last_percentage:
            self._last_percentage = percentage
            if percentage % 10 == 0 or percentage == 100:
                print(f"{self.description}: {percentage}% complete "
                      f"({self.current_step}/{self.total_steps})")
    
    def reset(self) -> None:
        """Reset progress tracker."""
        self.current_step = 0
        self._last_percentage = -1


def warn_and_continue(message: str, category: type = UserWarning) -> None:
    """
    Issue a warning and continue execution.
    
    Parameters
    ----------
    message : str
        Warning message
    category : type, optional
        Warning category (default: UserWarning)
    """
    warnings.warn(message, category)


class InvalidEstimate(Exception):
    """Exception raised when an estimator fails."""
    pass


def create_invalid_result(method: str, n: int) -> Dict[str, Any]:
    """
    Create a standard invalid result structure.
    
    Parameters
    ----------
    method : str
        Method name
    n : int
        Sample size
    
    Returns
    -------
    Dict[str, Any]
        Invalid result dictionary
    """
    return {
        "method": method,
        "tau_hat": np.nan,
        "treated_weights": np.zeros(n),
        "control_weights": np.zeros(n),
        "estimand_drift": np.nan,
        "valid": False,
    }


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