"""
Tests for the data generation module.
"""

import pytest
import numpy as np
from misspecification_tolerance.config import SimulationConfig, get_pilot_config
from misspecification_tolerance.data_generation import DataGenerator


def test_covariate_generation():
    """Test that covariates are generated with correct dimensions."""
    config = get_pilot_config()
    rng = np.random.default_rng(42)
    generator = DataGenerator(config, rng)
    
    X = generator.generate_covariates(n=100, rho=0.3)
    
    assert X.shape == (100, 6)
    assert np.allclose(np.mean(X, axis=0), 0, atol=0.5)
    assert np.allclose(np.std(X, axis=0), 1, atol=0.5)


def test_q_function_regimes():
    """Test that all q-function regimes work."""
    config = get_pilot_config()
    rng = np.random.default_rng(42)
    generator = DataGenerator(config, rng)
    
    X = generator.generate_covariates(n=10, rho=0.3)
    
    regimes = ["functional", "interaction", "combined", "omitted", "irrelevant"]
    
    for regime in regimes:
        qx = generator.q_function(X, regime)
        assert qx.shape == (10,)
        assert np.all(np.isfinite(qx))


def test_true_propensity():
    """Test true propensity score calculation."""
    config = get_pilot_config()
    rng = np.random.default_rng(42)
    generator = DataGenerator(config, rng)
    
    X = generator.generate_covariates(n=100, rho=0.3)
    a = generator.baseline_treatment_coefficients("functional")
    a0 = 0.0
    
    e0 = generator.true_propensity(X, a0, a, delta=0.5, q_regime="functional")
    
    assert e0.shape == (100,)
    assert np.all(e0 >= 0)
    assert np.all(e0 <= 1)


def test_intercept_calibration():
    """Test that intercept calibration achieves target prevalence."""
    config = get_pilot_config()
    rng = np.random.default_rng(42)
    generator = DataGenerator(config, rng)
    
    X_cal = generator.generate_covariates(n=10000, rho=0.3)
    a = generator.baseline_treatment_coefficients("functional")
    
    a0 = generator.calibrate_intercept(
        X_cal=X_cal,
        a=a,
        delta=0.5,
        q_regime="functional",
        pi_treat=0.35
    )
    
    # Check that calibration works
    e0 = generator.true_propensity(X_cal, a0, a, delta=0.5, q_regime="functional")
    assert np.abs(np.mean(e0) - 0.35) < 0.01


def test_dataset_generation():
    """Test full dataset generation."""
    config = get_pilot_config()
    rng = np.random.default_rng(42)
    generator = DataGenerator(config, rng)
    
    data = generator.generate_dataset(
        delta=0.5,
        q_regime="functional",
        outcome_model="linear",
        effect_type="constant",
        n=100,
    )
    
    # Check all keys exist
    expected_keys = [
        "X", "T", "Y", "Y0", "Y1", "tau_x", "e0",
        "att_sample", "ato_sample",
        "sigma_y", "sigma_tau_treated"
    ]
    for key in expected_keys:
        assert key in data
    
    # Check shapes
    assert data["X"].shape == (100, 6)
    assert data["T"].shape == (100,)
    assert data["Y"].shape == (100,)
    assert data["e0"].shape == (100,)
    
    # Check treatment prevalence
    pi_empirical = np.mean(data["T"])
    assert np.abs(pi_empirical - 0.35) < 0.15  # Allow some variation
    
    # Check ATT
    assert np.isfinite(data["att_sample"])
    
    # Check ATO
    assert np.isfinite(data["ato_sample"])


def test_outcome_models():
    """Test linear and nonlinear outcome models."""
    config = get_pilot_config()
    rng = np.random.default_rng(42)
    generator = DataGenerator(config, rng)
    
    X = generator.generate_covariates(n=100, rho=0.3)
    
    # Linear model
    m0_linear = generator.outcome_mean(X, "linear", s_b=1.0)
    assert m0_linear.shape == (100,)
    
    # Nonlinear model
    m0_nonlinear = generator.outcome_mean(X, "nonlinear", s_b=1.0)
    assert m0_nonlinear.shape == (100,)
    
    # Should be different
    assert not np.allclose(m0_linear, m0_nonlinear)


def test_treatment_effects():
    """Test treatment effect functions."""
    config = get_pilot_config()
    rng = np.random.default_rng(42)
    generator = DataGenerator(config, rng)
    
    X = generator.generate_covariates(n=100, rho=0.3)
    
    # Constant effect
    tau_const = generator.treatment_effect_function(X, "constant")
    assert np.all(tau_const == 1.0)
    
    # Linear effect
    tau_linear = generator.treatment_effect_function(X, "linear")
    assert tau_linear.shape == (100,)
    assert not np.all(tau_linear == 1.0)
    
    # Heterogeneous effect
    tau_hetero = generator.treatment_effect_function(X, "heterogeneous")
    assert tau_hetero.shape == (100,)
    assert not np.all(tau_hetero == 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])