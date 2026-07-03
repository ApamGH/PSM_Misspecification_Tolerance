"""
Configuration Module
===================

Defines the simulation configuration dataclass with all parameters
as specified in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import json
import yaml
import numpy as np


@dataclass
class SimulationConfig:
    """
    Configuration for the misspecification-tolerance simulation.
    
    All parameters are based on the paper's simulation design (Section 5).
    """
    
    # ============================================================
    # Sample and Monte Carlo Settings (Section 5: Sample size,
    # replications, and scenario structure)
    # ============================================================
    n: int = 1000  # Sample size
    R: int = 500   # Number of Monte Carlo replications
    n_jobs: int = 1  # Parallel jobs; 1 is safer for local machines
    
    # Covariate correlation and treatment prevalence
    rho: float = 0.30  # Equicorrelation among covariates
    pi_treat: float = 0.35  # Target treatment prevalence
    
    # Treatment-assignment and outcome prognostic strength
    s_a: float = 1.00  # Overlap severity (treatment assignment strength)
    s_b: float = 1.00  # Prognostic strength (outcome association)
    
    # Random noise in the outcome
    sigma: float = 1.00  # Standard deviation of outcome noise
    
    # ============================================================
    # Misspecification Grid (Section 5: Misspecification regimes)
    # ============================================================
    delta_grid: Tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.arange(0.0, 2.01, 0.10), 2))
    )
    
    # ============================================================
    # Simulation Regimes (Section 5: Misspecification regimes)
    # ============================================================
    q_regimes: Tuple[str, ...] = (
        "functional",   # q(X) = X1^2 + X2^2
        "interaction",  # q(X) = X1*X2 + X3*X4
        "combined",     # q(X) = functional + interaction
        "omitted",      # q(X) = X4 (omitted confounder)
        "irrelevant",   # q(X) = X5, X6 (predicts treatment only)
    )
    
    outcome_models: Tuple[str, ...] = (
        "linear",      # m0(X) = linear in X
        "nonlinear",   # m0(X) = linear + quadratic + interaction
    )
    
    treatment_effects: Tuple[str, ...] = (
        "constant",    # tau(X) = constant
        "linear",      # tau(X) = observed-covariate heterogeneity
    )
    
    # ============================================================
    # Methods Included (Section 4: Estimator-specific boundaries)
    # ============================================================
    methods: Tuple[str, ...] = (
        "PSM",              # Propensity Score Matching
        "FM",               # Full Matching
        "IPW",              # Inverse Probability Weighting
        "OW",               # Overlap Weighting
        "EB",               # Entropy Balancing
        "AIPW_correct",     # AIPW with correct outcome model
        "AIPW_restricted",  # AIPW with restricted outcome model
    )
    
    # ============================================================
    # Overlap and Effective Support Settings (Section 5:
    # Design diagnostics)
    # ============================================================
    eta_overlap: float = 0.05  # Overlap trimming threshold
    kappa_weight: float = 1e-5  # Effective weight threshold
    
    # ============================================================
    # Confidence Interval Settings (Section 5: Monte Carlo
    # performance measures)
    # ============================================================
    alpha: float = 0.05  # Significance level for 95% CI
    n_bootstrap: int = 1000  # Bootstrap replications for CI
    
    # ============================================================
    # Diagnostic Thresholds (Section 5: Diagnostic thresholds and
    # loss function)
    # ============================================================
    eps_bias_multiplier: float = 0.10  # epsilon_B = 0.10 * sigma_Y
    eps_rmse_multiplier: float = 0.20  # epsilon_R = 0.20 * sigma_Y
    eps_coverage: float = 0.025  # epsilon_C = 0.025 (coverage error)
    eps_fragility: float = 0.25  # epsilon_F = 0.25 (prognostic imbalance)
    eps_overlap: float = 0.10  # epsilon_O = 0.10 (overlap degradation)
    eps_drift_multiplier: float = 0.10  # epsilon_S = 0.10 * sigma_tau
    
    # ============================================================
    # Loss Function Weights (Section 4: Common diagnostic structure)
    # ============================================================
    # Default: equal weights across active components
    lambda_B: float = 1.0/6.0  # Bias weight
    lambda_R: float = 1.0/6.0  # RMSE weight
    lambda_C: float = 1.0/6.0  # Coverage error weight
    lambda_F: float = 1.0/6.0  # Fragility weight
    lambda_O: float = 1.0/6.0  # Overlap weight
    lambda_S: float = 1.0/6.0  # Estimand drift weight
    
    # ============================================================
    # Method-Specific Settings (Section 4)
    # ============================================================
    # PSM settings
    psm_neighbors: int = 1  # K = 1 (one-to-one matching)
    psm_caliper_multiplier: float = 0.20  # 0.20 * SD(logit(e))
    psm_with_replacement: bool = False  # Matching without replacement
    
    # Full matching settings
    fm_initial_subclasses: int = 20  # Initial number of subclasses
    fm_min_subclass_size: int = 5  # Minimum units per subclass
    
    # Entropy balancing settings
    eb_extended_balance: bool = True  # Use extended balance basis
    eb_maxiter: int = 1000  # Maximum iterations
    eb_tolerance: float = 1e-8  # Convergence tolerance
    
    # ============================================================
    # Scenario Exploration (Section 5: Sample size, replications,
    # and scenario structure)
    # ============================================================
    # Values for full factorial design
    # Keep defaults local-machine friendly. Broader factorial settings should
    # be activated deliberately through a separate sensitivity configuration.
    n_values: Tuple[int, ...] = (1000,)
    rho_values: Tuple[float, ...] = (0.30,)
    pi_treat_values: Tuple[float, ...] = (0.35,)
    s_a_values: Tuple[float, ...] = (1.00,)
    s_b_values: Tuple[float, ...] = (1.00,)
    
    # ============================================================
    # Calibration and Random Seed
    # ============================================================
    calibration_size: int = 100_000  # Size for intercept calibration
    random_seed: int = 20260627  # Random seed for reproducibility
    
    # ============================================================
    # Output Settings
    # ============================================================
    output_dir: str = "outputs"
    save_checkpoints: bool = True
    checkpoint_interval: int = 10  # Save checkpoint every N scenarios
    verbose: bool = True
    
    @classmethod
    def from_json(cls, path: str) -> "SimulationConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert lists to tuples for immutable fields
        tuple_fields = [
            'delta_grid', 'q_regimes', 'outcome_models',
            'treatment_effects', 'methods', 'n_values',
            'rho_values', 'pi_treat_values', 's_a_values', 's_b_values'
        ]
        for field in tuple_fields:
            if field in data and isinstance(data[field], list):
                data[field] = tuple(data[field])
        
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: str) -> "SimulationConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert lists to tuples for immutable fields
        tuple_fields = [
            'delta_grid', 'q_regimes', 'outcome_models',
            'treatment_effects', 'methods', 'n_values',
            'rho_values', 'pi_treat_values', 's_a_values', 's_b_values'
        ]
        for field in tuple_fields:
            if field in data and isinstance(data[field], list):
                data[field] = tuple(data[field])
        
        return cls(**data)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        data = self.__dict__.copy()
        # Convert tuples to lists for JSON serialization
        for key, value in data.items():
            if isinstance(value, tuple):
                data[key] = list(value)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        data = self.__dict__.copy()
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def get_scenario_dict(self) -> Dict[str, Any]:
        """
        Get scenario parameters as a dictionary.
        
        Returns only the parameters that define a scenario,
        excluding simulation-run settings.
        """
        scenario_params = {
            'n': self.n,
            'rho': self.rho,
            'pi_treat': self.pi_treat,
            's_a': self.s_a,
            's_b': self.s_b,
            'sigma': self.sigma,
        }
        return scenario_params
    
    def get_diagnostic_thresholds(self) -> Dict[str, float]:
        """Get diagnostic thresholds dictionary."""
        return {
            'eps_B': self.eps_bias_multiplier * self.sigma,
            'eps_R': self.eps_rmse_multiplier * self.sigma,
            'eps_C': self.eps_coverage,
            'eps_F': self.eps_fragility,
            'eps_O': self.eps_overlap,
            'eps_S': self.eps_drift_multiplier * self.sigma,  # Will be refined
        }
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss function weights."""
        return {
            'lambda_B': self.lambda_B,
            'lambda_R': self.lambda_R,
            'lambda_C': self.lambda_C,
            'lambda_F': self.lambda_F,
            'lambda_O': self.lambda_O,
            'lambda_S': self.lambda_S,
        }


# ============================================================
# Predefined Configurations
# ============================================================

def get_pilot_config() -> SimulationConfig:
    """Get pilot simulation configuration."""
    return SimulationConfig(
        n=100,
        R=50,
        n_jobs=1,  # Sequential for debugging
        rho=0.30,
        pi_treat=0.35,
        s_a=1.00,
        s_b=1.00,
        sigma=1.00,
        delta_grid=tuple(np.round(np.arange(0.0, 2.01, 0.20), 2)),
        q_regimes=("functional", "interaction"),
        outcome_models=("linear",),
        treatment_effects=("constant",),
        methods=("PSM", "IPW", "OW"),
        n_values=(100,),
        rho_values=(0.30,),
        pi_treat_values=(0.35,),
        s_a_values=(1.00,),
        s_b_values=(1.00,),
        output_dir="outputs/pilot",
        verbose=True,
    )


def get_main_config() -> SimulationConfig:
    """
    Get a local-machine main simulation configuration.

    This is deliberately not a full factorial design. It is aligned with the
    current manuscript objective and is more realistic for a 64-bit local
    machine, especially with entropy balancing included. Broader rho, sample
    size, prevalence, and overlap-strength sweeps should be run as sensitivity
    analyses after the main design is stable.
    """
    return SimulationConfig(
        n=1000,
        R=500,
        n_jobs=1,
        rho=0.30,
        pi_treat=0.35,
        s_a=1.00,
        s_b=1.00,
        sigma=1.00,
        delta_grid=tuple(np.round(np.arange(0.0, 2.01, 0.10), 2)),
        q_regimes=("functional", "interaction", "combined", "omitted", "irrelevant"),
        outcome_models=("linear", "nonlinear"),
        treatment_effects=("constant", "linear"),
        methods=("PSM", "FM", "IPW", "OW", "EB", "AIPW_correct", "AIPW_restricted"),
        n_values=(1000,),
        rho_values=(0.30,),
        pi_treat_values=(0.35,),
        s_a_values=(1.00,),
        s_b_values=(1.00,),
        calibration_size=100_000,
        output_dir="outputs/main",
        verbose=True,
    )


def get_sensitivity_config() -> SimulationConfig:
    """Get sensitivity analysis configuration."""
    return SimulationConfig(
        n=2500,
        R=2000,
        n_jobs=-1,
        rho=0.60,
        pi_treat=0.50,
        s_a=0.50,
        s_b=1.50,
        sigma=1.00,
        delta_grid=tuple(np.round(np.arange(0.0, 2.01, 0.05), 2)),  # Finer grid
        q_regimes=("combined", "omitted"),
        outcome_models=("linear", "nonlinear"),
        treatment_effects=("constant", "heterogeneous"),
        methods=("PSM", "OW", "EB", "AIPW_correct"),
        n_values=(2500, 5000),
        rho_values=(0.0, 0.60),
        pi_treat_values=(0.20, 0.50),
        s_a_values=(0.50, 1.50),
        s_b_values=(0.50, 1.50),
        output_dir="outputs/sensitivity",
        verbose=True,
    )