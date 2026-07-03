"""
Data Generation Module
======================

Implements the data-generating process described in Section 5 of the paper.

Key Features:
- 6 covariates with equicorrelation structure (p = 6)
- True propensity score with controlled misspecification
- 5 misspecification regimes
- Outcome generation (linear and nonlinear)
- Treatment effect generation (constant, linear, heterogeneous)
- Intercept calibration for treatment prevalence
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.optimize import brentq
import warnings

from .config import SimulationConfig
from .utils import logistic, warn_and_continue


class DataGenerator:
    """
    Data generator for the misspecification-tolerance simulation.
    
    Generates datasets with controlled treatment assignment misspecification,
    following the design in Section 5 of the paper.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    rng : np.random.Generator
        Random number generator
    """
    
    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        # Cache calibration samples by rho so intercept calibration uses
        # the same covariate correlation structure as the scenario being run.
        self._calibration_cache: Dict[float, np.ndarray] = {}

    def _get_calibration_X(self, rho: float) -> np.ndarray:
        """Return a calibration sample generated at the scenario-specific rho."""
        key = float(rho)
        if key not in self._calibration_cache:
            self._calibration_cache[key] = self.generate_covariates(
                n=self.config.calibration_size,
                rho=rho,
            )
        return self._calibration_cache[key]
    
    def generate_covariates(self, n: int, rho: float) -> np.ndarray:
        """
        Generate multivariate normal covariates.
        
        Following Section 5: p = 6 covariates with equicorrelation.
        
        Parameters
        ----------
        n : int
            Number of samples
        rho : float
            Equicorrelation coefficient
        
        Returns
        -------
        np.ndarray
            Covariate matrix of shape (n, 6)
        """
        p = 6  # Fixed at 6 covariates as per paper
        
        # Equicorrelation covariance matrix
        sigma = (1.0 - rho) * np.eye(p) + rho * np.ones((p, p))
        
        return self.rng.multivariate_normal(
            mean=np.zeros(p),
            cov=sigma,
            size=n
        )
    
    def q_function(self, X: np.ndarray, regime: str) -> np.ndarray:
        """
        Misspecification-generating function q(X).
        
        Implements the 5 misspecification regimes from Section 5:
        1. functional: q(X) = X1^2 + X2^2
        2. interaction: q(X) = X1*X2 + X3*X4
        3. combined: q(X) = functional + interaction
        4. omitted: q(X) = X4 (omitted confounder)
        5. irrelevant: q(X) = X5, X6 (predicts treatment only)
        
        Parameters
        ----------
        X : np.ndarray
            Covariate matrix (n, 6)
        regime : str
            Misspecification regime name
        
        Returns
        -------
        np.ndarray
            q(X) values of shape (n,)
        
        Raises
        ------
        ValueError
            If regime is unknown
        """
        # Extract covariates
        x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
        
        if regime == "functional":
            # Functional-form misspecification: quadratic terms
            return 0.60 * x1**2 - 0.40 * x2**2
        
        elif regime == "interaction":
            # Interaction misspecification
            return 0.70 * x1 * x2 - 0.50 * x3 * x4
        
        elif regime == "combined":
            # Combined nonlinear and interaction misspecification
            return (
                0.50 * x1**2 
                - 0.30 * x2**2 
                + 0.50 * x1 * x2 
                - 0.40 * x3 * x4
            )
        
        elif regime == "omitted":
            # Omitted confounder: X4 affects treatment and outcome
            return 0.80 * x4
        
        elif regime == "irrelevant":
            # Treatment-predictive but outcome-irrelevant misspecification.
            # X5 and X6 do not predict the outcome, but nonlinear treatment
            # assignment structure in these variables is omitted from the
            # linear working propensity model.
            return 0.60 * x5**2 - 0.40 * x6**2 + 0.30 * x5 * x6
        
        else:
            raise ValueError(f"Unknown q regime: {regime}")
    
    def baseline_treatment_coefficients(self, q_regime: str) -> np.ndarray:
        """
        Baseline treatment coefficients.
        
        For the omitted-confounder regime, X4 is removed from the baseline
        treatment component so that delta=0 remains correctly specified
        by the working propensity model that excludes X4.
        
        Parameters
        ----------
        q_regime : str
            Misspecification regime
        
        Returns
        -------
        np.ndarray
            Coefficient vector of length 6
        """
        # Base coefficients for 6 covariates
        base = np.array([0.60, -0.50, 0.40, -0.35, 0.25, -0.20])
        
        if q_regime == "omitted":
            # Remove X4 from baseline treatment component
            base[3] = 0.0
        
        return self.config.s_a * base
    
    def _true_linear_predictor(
        self,
        X: np.ndarray,
        a0: float,
        a: np.ndarray,
        delta: float,
        q_regime: str
    ) -> np.ndarray:
        """
        Compute the true treatment assignment linear predictor.
        
        eta_0(X; delta) = a0 + s_a * (a'X) + delta * q(X)
        
        Parameters
        ----------
        X : np.ndarray
            Covariate matrix (n, 6)
        a0 : float
            Intercept
        a : np.ndarray
            Treatment coefficients
        delta : float
            Misspecification intensity
        q_regime : str
            Misspecification regime
        
        Returns
        -------
        np.ndarray
            Linear predictor values of shape (n,)
        """
        linear_part = X @ a
        qx = self.q_function(X, q_regime)
        
        return a0 + linear_part + delta * qx
    
    def true_propensity(
        self,
        X: np.ndarray,
        a0: float,
        a: np.ndarray,
        delta: float,
        q_regime: str
    ) -> np.ndarray:
        """
        Compute the true propensity score e0(X; delta).
        
        e0(X; delta) = logistic(a0 + s_a * a'X + delta * q(X))
        
        Parameters
        ----------
        X : np.ndarray
            Covariate matrix (n, 6)
        a0 : float
            Intercept
        a : np.ndarray
            Treatment coefficients
        delta : float
            Misspecification intensity
        q_regime : str
            Misspecification regime
        
        Returns
        -------
        np.ndarray
            True propensity scores of shape (n,)
        """
        eta = self._true_linear_predictor(X, a0, a, delta, q_regime)
        return logistic(eta)
    
    def calibrate_intercept(
        self,
        X_cal: np.ndarray,
        a: np.ndarray,
        delta: float,
        q_regime: str,
        pi_treat: float
    ) -> float:
        """
        Calibrate the treatment intercept to achieve target treatment prevalence.
        
        Solves for a0 such that:
        E[logistic(a0 + s_a * a'X + delta * q(X))] = pi_treat
        
        Parameters
        ----------
        X_cal : np.ndarray
            Calibration sample
        a : np.ndarray
            Treatment coefficients
        delta : float
            Misspecification intensity
        q_regime : str
            Misspecification regime
        pi_treat : float
            Target treatment prevalence
        
        Returns
        -------
        float
            Calibrated intercept
        """
        qx = self.q_function(X_cal, q_regime)
        linear_part = X_cal @ a + delta * qx
        
        def root_function(a0: float) -> float:
            return float(np.mean(logistic(a0 + linear_part)) - pi_treat)
        
        try:
            # Use Brent's method for root finding
            return brentq(root_function, -20.0, 20.0, maxiter=200)
        except ValueError:
            # Fallback: use logit of prevalence
            warn_and_continue(
                f"Intercept calibration failed in [-20, 20] for delta={delta}. "
                "Using approximate logit-prevalence intercept."
            )
            return float(np.log(pi_treat / (1.0 - pi_treat)))
    
    def outcome_mean(
        self,
        X: np.ndarray,
        model: str,
        s_b: float
    ) -> np.ndarray:
        """
        Generate the untreated outcome regression m0(X).
        
        Two models from Section 5:
        - linear: m0(X) = b0 + s_b * b'X
        - nonlinear: m0(X) = linear + b7*X1^2 + b8*X2^2 + b9*X1*X2
        
        X4 is always outcome-prognostic, which is crucial for the
        omitted-confounder regime.
        
        Parameters
        ----------
        X : np.ndarray
            Covariate matrix (n, 6)
        model : str
            Outcome model type ('linear' or 'nonlinear')
        s_b : float
            Prognostic strength
        
        Returns
        -------
        np.ndarray
            Outcome mean values of shape (n,)
        
        Raises
        ------
        ValueError
            If model is unknown
        """
        # Coefficients for 6 covariates. X5 and X6 are set to zero so
        # the "irrelevant" regime can represent treatment-predictive but
        # outcome-irrelevant structure.
        b = np.array([0.70, -0.60, 0.50, -0.40, 0.00, 0.00])
        b_scaled = s_b * b
        
        # Extract covariates
        x1, x2 = X[:, 0], X[:, 1]
        
        # Linear component
        mu = 0.50 + X @ b_scaled
        
        if model == "linear":
            return mu
        
        elif model == "nonlinear":
            # Add nonlinear terms
            return mu + 0.40 * x1**2 - 0.30 * x1 * x2
        
        else:
            raise ValueError(f"Unknown outcome model: {model}")
    
    def treatment_effect_function(
        self,
        X: np.ndarray,
        effect_type: str
    ) -> np.ndarray:
        """
        Generate unit-level treatment effects tau(X).
        
        Three settings from Section 5:
        - constant: tau(X) = 1.0
        - linear: tau(X) = 1.0 + 0.25*X1 - 0.20*X2 + 0.15*X3
        - heterogeneous: tau(X) = linear + 0.10*X1*X2 - 0.15*X3*X4
        
        Parameters
        ----------
        X : np.ndarray
            Covariate matrix (n, 6)
        effect_type : str
            Treatment effect type
        
        Returns
        -------
        np.ndarray
            Treatment effect values of shape (n,)
        
        Raises
        ------
        ValueError
            If effect_type is unknown
        """
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        
        if effect_type == "constant":
            return np.full(X.shape[0], 1.0)
        
        elif effect_type == "linear":
            return 1.0 + 0.25 * x1 - 0.20 * x2 + 0.15 * x3
        
        elif effect_type == "heterogeneous":
            return (
                1.0 
                + 0.25 * x1 
                - 0.20 * x2 
                + 0.15 * x3
                + 0.10 * x1 * x2 
                - 0.15 * x3 * x4
            )
        
        else:
            raise ValueError(f"Unknown treatment effect type: {effect_type}")
    
    def compute_ato(
        self,
        tau_x: np.ndarray,
        e0: np.ndarray
    ) -> float:
        """
        Compute the Average Treatment Effect in the Overlap population (ATO).
        
        tau_ATO = sum(tau * e0 * (1-e0)) / sum(e0 * (1-e0))
        
        Parameters
        ----------
        tau_x : np.ndarray
            Individual treatment effects
        e0 : np.ndarray
            True propensity scores
        
        Returns
        -------
        float
            ATO estimate
        """
        weights = e0 * (1.0 - e0)
        
        numerator = np.sum(weights * tau_x)
        denominator = np.sum(weights)
        
        if denominator <= 0:
            return np.nan
        
        return float(numerator / denominator)
    
    def generate_dataset(
        self,
        delta: float,
        q_regime: str,
        outcome_model: str,
        effect_type: str,
        n: Optional[int] = None,
        rho: Optional[float] = None,
        pi_treat: Optional[float] = None,
        s_a: Optional[float] = None,
        s_b: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate one complete simulated dataset.
        
        This is the main data generation function that creates:
        - Covariates X (n x 6)
        - Treatment assignment T
        - Potential outcomes Y0, Y1
        - Observed outcome Y
        - True propensity scores e0
        - Individual treatment effects tau_x
        - True estimands (ATT, ATO)
        
        Parameters
        ----------
        delta : float
            Misspecification intensity
        q_regime : str
            Misspecification regime
        outcome_model : str
            Outcome model type
        effect_type : str
            Treatment effect type
        n : int, optional
            Sample size (overrides config)
        rho : float, optional
            Covariate correlation (overrides config)
        pi_treat : float, optional
            Treatment prevalence (overrides config)
        s_a : float, optional
            Overlap severity (overrides config)
        s_b : float, optional
            Prognostic strength (overrides config)
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing all generated data and metadata
        
        Raises
        ------
        RuntimeError
            If generated dataset has zero treated or zero controls
        """
        # Use provided parameters or fall back to config
        n = n if n is not None else self.config.n
        rho = rho if rho is not None else self.config.rho
        pi_treat = pi_treat if pi_treat is not None else self.config.pi_treat
        s_a = s_a if s_a is not None else self.config.s_a
        s_b = s_b if s_b is not None else self.config.s_b
        sigma = self.config.sigma
        
        # 1. Generate covariates
        X = self.generate_covariates(n, rho)
        
        # 2. Generate treatment assignment
        a = self.baseline_treatment_coefficients(q_regime) * (s_a / self.config.s_a)
        
        # Calibrate intercept using a calibration sample generated with
        # the scenario-specific rho.
        X_cal = self._get_calibration_X(rho)
        a0 = self.calibrate_intercept(
            X_cal=X_cal,
            a=a,
            delta=delta,
            q_regime=q_regime,
            pi_treat=pi_treat,
        )
        
        # True propensity scores
        e0 = self.true_propensity(
            X=X,
            a0=a0,
            a=a,
            delta=delta,
            q_regime=q_regime,
        )
        
        # Treatment assignment
        U = self.rng.uniform(0.0, 1.0, size=n)
        T = (U <= e0).astype(int)
        
        # Check for sufficient treated/control units
        n1 = int(np.sum(T))
        n0 = int(n - n1)
        
        if n1 == 0 or n0 == 0:
            raise RuntimeError(
                f"Generated dataset has zero treated (n1={n1}) or zero controls (n0={n0}). "
                f"Try adjusting pi_treat or s_a."
            )
        
        # 3. Generate outcomes
        # Untreated outcome mean
        m0 = self.outcome_mean(X, outcome_model, s_b)
        
        # Treatment effects
        tau_x = self.treatment_effect_function(X, effect_type)
        
        # Outcome noise
        eps = self.rng.normal(0.0, sigma, size=n)
        
        # Potential outcomes
        Y0 = m0 + eps
        Y1 = Y0 + tau_x
        
        # Observed outcome
        Y = Y0 + T * tau_x
        
        # 4. Compute true estimands
        att_sample = float(np.mean(tau_x[T == 1]))
        ato_sample = self.compute_ato(tau_x, e0)
        
        # 5. Additional metadata
        sigma_y = float(np.std(Y, ddof=1))
        sigma_tau_treated = (
            float(np.std(tau_x[T == 1], ddof=1))
            if n1 > 1
            else 0.0
        )
        
        return {
            # Data
            "X": X,
            "T": T,
            "Y": Y,
            "Y0": Y0,
            "Y1": Y1,
            "tau_x": tau_x,
            "e0": e0,
            
            # True estimands
            "att_sample": att_sample,
            "ato_sample": ato_sample,
            
            # Metadata
            "sigma_y": sigma_y,
            "sigma_tau_treated": sigma_tau_treated,
            
            # Parameters (for reference)
            "n": n,
            "rho": rho,
            "delta": delta,
            "q_regime": q_regime,
            "outcome_model": outcome_model,
            "effect_type": effect_type,
            "a0": a0,
            "a": a,
            "s_a": s_a,
            "s_b": s_b,
            "pi_treat": pi_treat,
        }
    
    def generate_multiple_datasets(
        self,
        delta: float,
        q_regime: str,
        outcome_model: str,
        effect_type: str,
        R: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate multiple datasets for Monte Carlo replications.
        
        Parameters
        ----------
        delta : float
            Misspecification intensity
        q_regime : str
            Misspecification regime
        outcome_model : str
            Outcome model type
        effect_type : str
            Treatment effect type
        R : int, optional
            Number of replications (overrides config)
        **kwargs
            Additional arguments passed to generate_dataset
        
        Returns
        -------
        List[Dict[str, np.ndarray]]
            List of generated datasets
        """
        R = R if R is not None else self.config.R
        
        datasets = []
        for _ in range(R):
            try:
                data = self.generate_dataset(
                    delta=delta,
                    q_regime=q_regime,
                    outcome_model=outcome_model,
                    effect_type=effect_type,
                    **kwargs
                )
                datasets.append(data)
            except RuntimeError as e:
                warn_and_continue(f"Dataset generation failed: {e}")
                continue
        
        if len(datasets) == 0:
            raise RuntimeError("All dataset generations failed.")
        
        return datasets


# ============================================================
# Helper Functions for Working Propensity and Outcome Models
# ============================================================

def make_propensity_design(X: np.ndarray, q_regime: str) -> np.ndarray:
    """
    Construct the working propensity-score design matrix.
    
    For the omitted-confounder regime:
        X4 is excluded from the working treatment model.
    
    For all other regimes:
        all six observed covariates are included linearly.
    
    Parameters
    ----------
    X : np.ndarray
        Covariate matrix (n, 6)
    q_regime : str
        Misspecification regime
    
    Returns
    -------
    np.ndarray
        Design matrix for working propensity model
    """
    if q_regime == "omitted":
        # Exclude X4 (the omitted confounder)
        return X[:, [0, 1, 2, 4, 5]]  # X1, X2, X3, X5, X6
    else:
        # Include all covariates
        return X


def make_outcome_design(X: np.ndarray, extended: bool = False) -> np.ndarray:
    """
    Construct outcome regression design matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Covariate matrix (n, 6)
    extended : bool, optional
        If True, adds X1^2 and X1*X2 for nonlinear outcome model
    
    Returns
    -------
    np.ndarray
        Design matrix for outcome regression
    """
    if not extended:
        return X
    
    x1, x2 = X[:, 0], X[:, 1]
    
    return np.column_stack([
        X,
        x1**2,
        x1 * x2,
    ])


def fit_propensity(
    X: np.ndarray,
    T: np.ndarray,
    q_regime: str,
    max_iter: int = 1000
) -> np.ndarray:
    """
    Fit the deliberately restricted working logistic propensity model.
    
    Parameters
    ----------
    X : np.ndarray
        Covariate matrix (n, 6)
    T : np.ndarray
        Treatment indicator
    q_regime : str
        Misspecification regime
    max_iter : int, optional
        Maximum iterations for logistic regression
    
    Returns
    -------
    np.ndarray
        Estimated propensity scores
    """
    from sklearn.linear_model import LogisticRegression
    
    X_work = make_propensity_design(X, q_regime)
    
    # Use a very weak ridge penalty for numerical stability and to avoid
    # scikit-learn version differences around penalty=None/"none".
    model = LogisticRegression(
        penalty="l2",
        C=1e9,
        solver="lbfgs",
        max_iter=max_iter,
    )
    
    model.fit(X_work, T)
    ehat = model.predict_proba(X_work)[:, 1]
    
    # Clip to avoid extreme values
    return np.clip(ehat, 1e-6, 1.0 - 1e-6)


def fit_outcome_control(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    extended: bool = False
) -> np.ndarray:
    """
    Fit outcome model m0(X) using controls only.
    
    Parameters
    ----------
    X : np.ndarray
        Covariate matrix (n, 6)
    T : np.ndarray
        Treatment indicator
    Y : np.ndarray
        Observed outcome
    extended : bool, optional
        If True, uses extended design with nonlinear terms
    
    Returns
    -------
    np.ndarray
        Predicted outcome values
    """
    from sklearn.linear_model import LinearRegression
    
    X_design = make_outcome_design(X, extended=extended)
    control = T == 0
    
    model = LinearRegression()
    model.fit(X_design[control], Y[control])
    
    return model.predict(X_design)