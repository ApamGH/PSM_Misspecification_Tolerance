"""
Tests for the diagnostics module.
"""

import pytest
import numpy as np
from misspecification_tolerance.config import get_pilot_config
from misspecification_tolerance.diagnostics import (
    h_basis,
    design_fragility,
    overlap_degradation,
    estimand_drift,
    diagnostic_loss_from_metrics,
    aggregate_replication_results,
    extract_tolerance_boundary,
)


def test_h_basis():
    """Test prognostic basis generation."""
    X = np.random.randn(10, 6)
    
    # Basic basis
    H_basic = h_basis(X, extended=False)
    assert H_basic.shape == (10, 6)
    
    # Extended basis
    H_extended = h_basis(X, extended=True)
    # 6 original + 4 quadratic + 4 interactions + 2 additional = 16
    assert H_extended.shape == (10, 16)


def test_design_fragility():
    """Test design fragility computation."""
    X = np.random.randn(100, 6)
    treated_weights = np.zeros(100)
    control_weights = np.zeros(100)
    
    # First 50 units treated, last 50 control
    treated_weights[:50] = 1.0 / 50
    control_weights[50:] = 1.0 / 50
    
    fragility = design_fragility(X, treated_weights, control_weights)
    
    assert np.isfinite(fragility)
    assert fragility >= 0


def test_overlap_degradation():
    """Test overlap degradation computation."""
    e0 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    ehat = np.array([0.15, 0.35, 0.45, 0.65, 0.85])
    
    degradation = overlap_degradation(e0, ehat, eta=0.05)
    
    assert np.isfinite(degradation)
    assert 0 <= degradation <= 1


def test_estimand_drift():
    """Test estimand drift computation."""
    tau_x = np.array([1.0, 1.5, 2.0, 0.5, 1.0])
    T = np.array([1, 1, 1, 0, 0])
    treated_weights = np.array([0.2, 0.3, 0.5, 0.0, 0.0])
    
    drift = estimand_drift(treated_weights, tau_x, T)
    
    assert np.isfinite(drift)
    # Expected: (0.2*1.0 + 0.3*1.5 + 0.5*2.0) - mean([1.0, 1.5, 2.0])
    # = (0.2 + 0.45 + 1.0) - 1.5 = 1.65 - 1.5 = 0.15
    assert np.abs(drift - 0.15) < 1e-10


def test_diagnostic_loss():
    """Test diagnostic loss computation."""
    thresholds = {
        'eps_B': 0.1,
        'eps_R': 0.2,
        'eps_C': 0.025,
        'eps_F': 0.25,
        'eps_O': 0.10,
        'eps_S': 0.10,
    }
    weights = {
        'lambda_B': 1.0/6.0,
        'lambda_R': 1.0/6.0,
        'lambda_C': 1.0/6.0,
        'lambda_F': 1.0/6.0,
        'lambda_O': 1.0/6.0,
        'lambda_S': 1.0/6.0,
    }
    
    metrics = {
        'bias': 0.05,
        'rmse': 0.10,
        'coverage_error': 0.01,
        'fragility': 0.12,
        'overlap_degradation': 0.04,
        'estimand_drift': 0.03,
    }
    
    loss = diagnostic_loss_from_metrics(
        metrics=metrics,
        thresholds=thresholds,
        weights=weights,
        is_constant_effect=False
    )
    
    expected = (0.05/0.1 + 0.10/0.2 + 0.01/0.025 + 0.12/0.25 + 0.04/0.10 + 0.03/0.10) / 6
    expected = (0.5 + 0.5 + 0.4 + 0.48 + 0.4 + 0.3) / 6
    expected = 2.58 / 6
    expected = 0.43
    
    assert np.abs(loss - expected) < 0.01


def test_aggregate_replication_results():
    """Test aggregation of replication results."""
    config = get_pilot_config()
    
    # Create some mock results
    results = []
    for i in range(10):
        result = {
            'tau_hat': 1.0 + np.random.randn() * 0.1,
            'target': 1.0,
            'coverage': 0.95 if np.random.rand() > 0.05 else 0.0,
            'fragility': 0.1 + np.random.rand() * 0.1,
            'overlap_degradation': 0.05 + np.random.rand() * 0.05,
            'estimand_drift': 0.02 + np.random.rand() * 0.03,
            'valid': True,
            'effect_type': 'constant',
        }
        results.append(result)
    
    aggregated = aggregate_replication_results(results, config)
    
    assert 'bias' in aggregated
    assert 'rmse' in aggregated
    assert 'coverage' in aggregated
    assert 'fragility' in aggregated
    assert 'overlap_degradation' in aggregated
    assert 'estimand_drift' in aggregated
    assert 'loss' in aggregated
    assert np.isfinite(aggregated['loss'])


def test_extract_tolerance_boundary():
    """Test tolerance boundary extraction."""
    import pandas as pd
    
    # Create mock summary data
    data = []
    for method in ['PSM', 'IPW', 'OW']:
        for delta in [0.0, 0.5, 1.0, 1.5, 2.0]:
            loss = 0.5 + delta * 0.3 if method == 'PSM' else 0.3 + delta * 0.2
            if method == 'PSM' and delta >= 1.5:
                loss = 1.2  # Fail
            elif method == 'IPW' and delta >= 2.0:
                loss = 1.1  # Fail at 2.0
            data.append({
                'method': method,
                'delta': delta,
                'loss': loss,
                'scenario': 'S1',
            })
    
    df = pd.DataFrame(data)
    boundaries, ratios = extract_tolerance_boundary(df)
    
    assert not boundaries.empty
    assert 'delta_star_conservative' in boundaries.columns
    assert 'PSM' in boundaries['method'].values
    
    # Check ratios
    if not ratios.empty:
        assert 'relative_tolerance' in ratios.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])