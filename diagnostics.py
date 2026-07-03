"""
Diagnostics Module
==================

Implements diagnostic functions for the misspecification-tolerance framework:

1. Design Fragility:
   F_m(delta) = ||Delta_{h,m}(delta)||_2

2. Effective-Support / Overlap Degradation:
   O_m(delta; eta) based on method-induced effective support.

3. Estimand Drift:
   S_m(delta) = tau_m^(1)(delta) - tau_ATT(delta)

4. Coverage Error:
   Q_m(delta) = max{0, 0.95 - C_m(delta)}

5. Diagnostic Loss:
   L_m(delta) = sum(lambda_j * component_j / threshold_j)

Important implementation detail:
- Loss-only tolerability is retained as `tolerable_loss_only`.
- Final tolerability also requires estimator validity:
      tolerable = tolerable_loss_only and valid_rate >= 0.90
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .utils import standardize_basis, weighted_mean


# ============================================================
# 1. Prognostic Basis h(X)
# ============================================================

def h_basis(X: np.ndarray, extended: bool = True) -> np.ndarray:
    """
    Prognostic basis h(X) for design-fragility diagnostics.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix with six columns.
    extended : bool, optional
        If True, include nonlinear and interaction terms.

    Returns
    -------
    np.ndarray
        Basis matrix.
    """
    if not extended:
        return X

    x1, x2, x3, x4, x5, x6 = (
        X[:, 0],
        X[:, 1],
        X[:, 2],
        X[:, 3],
        X[:, 4],
        X[:, 5],
    )

    return np.column_stack(
        [
            # Original covariates
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,

            # Quadratic terms
            x1**2,
            x2**2,
            x3**2,
            x4**2,

            # Interaction terms
            x1 * x2,
            x1 * x3,
            x2 * x3,
            x3 * x4,

            # Additional prognostic functions
            np.exp(-0.5 * x1**2),
            np.sin(x2),
        ]
    )


def design_fragility(
    X: np.ndarray,
    treated_weights: np.ndarray,
    control_weights: np.ndarray,
    extended: bool = True,
) -> float:
    """
    Compute prognostic design fragility.

    F_m(delta) = || mean_h_treated - mean_h_control ||_2 / sqrt(q)

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix.
    treated_weights : np.ndarray
        Weights for treated units.
    control_weights : np.ndarray
        Weights for control units.
    extended : bool, optional
        Whether to use the extended prognostic basis.

    Returns
    -------
    float
        Design fragility value.
    """
    if X is None or len(X) == 0:
        return np.nan

    if np.sum(treated_weights) <= 0 or np.sum(control_weights) <= 0:
        return np.nan

    H = h_basis(X, extended=extended)
    H_std = standardize_basis(H)

    q = H_std.shape[1]
    if q <= 0:
        return np.nan

    treated_mean = weighted_mean(H_std, treated_weights)
    control_mean = weighted_mean(H_std, control_weights)

    if not np.all(np.isfinite(treated_mean)) or not np.all(np.isfinite(control_mean)):
        return np.nan

    diff = treated_mean - control_mean
    fragility = np.linalg.norm(diff) / np.sqrt(q)

    return float(fragility)


# ============================================================
# 2. Overlap and Effective-Support Degradation
# ============================================================

def overlap_degradation(
    e0: np.ndarray,
    ehat: np.ndarray,
    eta: float = 0.05,
) -> float:
    """
    Compute score-overlap disagreement between true and working propensity scores.

    This is retained mainly as a descriptive diagnostic. The main loss uses
    method-specific effective-support degradation, which is later baseline-adjusted.

    Parameters
    ----------
    e0 : np.ndarray
        True propensity scores.
    ehat : np.ndarray
        Estimated propensity scores.
    eta : float, optional
        Overlap threshold.

    Returns
    -------
    float
        Proportion of observations classified differently between true and
        working overlap regions.
    """
    if e0 is None or ehat is None:
        return np.nan

    if len(e0) != len(ehat):
        return np.nan

    true_overlap = ((e0 >= eta) & (e0 <= 1.0 - eta)).astype(int)
    working_overlap = ((ehat >= eta) & (ehat <= 1.0 - eta)).astype(int)

    symmetric_diff = np.abs(true_overlap - working_overlap)

    return float(np.mean(symmetric_diff))


def effective_support_degradation(
    e0: np.ndarray,
    treated_weights: np.ndarray,
    control_weights: np.ndarray,
    eta: float = 0.05,
    kappa: float = 1e-5,
) -> float:
    """
    Compute method-specific effective-support degradation.

    The raw value compares the true overlap region with the effective support
    retained by the method's weights. The diagnostic loss should use the
    baseline-adjusted version computed during aggregation:
        max(0, O_raw(delta) - O_raw(delta=0)).

    Parameters
    ----------
    e0 : np.ndarray
        True propensity scores.
    treated_weights : np.ndarray
        Treated weights.
    control_weights : np.ndarray
        Control weights.
    eta : float, optional
        True-overlap threshold.
    kappa : float, optional
        Effective-weight threshold.

    Returns
    -------
    float
        Effective-support discrepancy.
    """
    if e0 is None:
        return np.nan

    if treated_weights is None or control_weights is None:
        return np.nan

    if len(e0) != len(treated_weights) or len(e0) != len(control_weights):
        return np.nan

    true_overlap = ((e0 >= eta) & (e0 <= 1.0 - eta)).astype(int)

    total_weights = treated_weights + control_weights
    effective_support = (total_weights > kappa).astype(int)

    return float(np.mean(np.abs(true_overlap - effective_support)))


# ============================================================
# 3. Estimand Drift
# ============================================================

def estimand_drift(
    treated_weights: np.ndarray,
    tau_x: np.ndarray,
    T: np.ndarray,
) -> float:
    """
    Compute method-induced estimand drift.

    S_m(delta) = tau_m^(1)(delta) - tau_ATT(delta)

    Parameters
    ----------
    treated_weights : np.ndarray
        Treated weights.
    tau_x : np.ndarray
        Individual treatment effects.
    T : np.ndarray
        Treatment indicator.

    Returns
    -------
    float
        Estimand drift.
    """
    if tau_x is None or T is None or treated_weights is None:
        return np.nan

    if len(tau_x) != len(T) or len(tau_x) != len(treated_weights):
        return np.nan

    if np.sum(treated_weights) <= 0:
        return np.nan

    if np.sum(T == 1) <= 0:
        return np.nan

    tau_method = np.sum(treated_weights * tau_x)
    tau_att = np.mean(tau_x[T == 1])

    if not np.isfinite(tau_method) or not np.isfinite(tau_att):
        return np.nan

    return float(tau_method - tau_att)


# ============================================================
# 4. Coverage
# ============================================================

def coverage_error(
    lower: float,
    upper: float,
    target: float,
) -> float:
    """
    Return whether a confidence interval covers the target.

    The returned value is a coverage indicator, not the final coverage penalty.
    Undercoverage-only penalty is computed during aggregation as:
        max(0, nominal_coverage - empirical_coverage)

    Parameters
    ----------
    lower : float
        Lower confidence limit.
    upper : float
        Upper confidence limit.
    target : float
        True target value.

    Returns
    -------
    float
        1.0 if covered, 0.0 if not covered, NaN if invalid.
    """
    if not np.isfinite(lower) or not np.isfinite(upper) or not np.isfinite(target):
        return np.nan

    return 1.0 if lower <= target <= upper else 0.0


# ============================================================
# 5. Diagnostic Loss
# ============================================================

@dataclass
class DiagnosticComponents:
    """Container for diagnostic-loss components."""

    bias: float
    rmse: float
    coverage_error: float
    fragility: float
    overlap_degradation: float
    estimand_drift: float

    def to_dict(self) -> Dict[str, float]:
        """Convert components to dictionary."""
        return {
            "bias": self.bias,
            "rmse": self.rmse,
            "coverage_error": self.coverage_error,
            "fragility": self.fragility,
            "overlap_degradation": self.overlap_degradation,
            "estimand_drift": self.estimand_drift,
        }


def compute_diagnostic_loss(
    components: DiagnosticComponents,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
    is_constant_effect: bool = False,
) -> float:
    """
    Compute diagnostic loss L_m(delta).

    For constant treatment effects, the estimand-drift component is inactive
    and its weight is redistributed across the five active components.

    Parameters
    ----------
    components : DiagnosticComponents
        Diagnostic components.
    thresholds : Dict[str, float]
        Component thresholds.
    weights : Dict[str, float]
        Component weights.
    is_constant_effect : bool, optional
        Whether treatment effects are constant.

    Returns
    -------
    float
        Diagnostic loss.
    """
    def safe_scaled(value: float, threshold: float, absolute: bool = False) -> float:
        if not np.isfinite(value):
            return np.nan

        if not np.isfinite(threshold) or threshold <= 0:
            return np.nan

        value = abs(value) if absolute else value

        return float(value / threshold)

    scaled_B = safe_scaled(
        components.bias,
        thresholds.get("eps_B", np.nan),
        absolute=True,
    )
    scaled_R = safe_scaled(
        components.rmse,
        thresholds.get("eps_R", np.nan),
        absolute=False,
    )
    scaled_C = safe_scaled(
        components.coverage_error,
        thresholds.get("eps_C", np.nan),
        absolute=False,
    )
    scaled_F = safe_scaled(
        components.fragility,
        thresholds.get("eps_F", np.nan),
        absolute=False,
    )
    scaled_O = safe_scaled(
        components.overlap_degradation,
        thresholds.get("eps_O", np.nan),
        absolute=False,
    )

    if is_constant_effect:
        scaled_S = 0.0
    else:
        scaled_S = safe_scaled(
            components.estimand_drift,
            thresholds.get("eps_S", np.nan),
            absolute=True,
        )

    weight_B = weights.get("lambda_B", 1.0 / 6.0)
    weight_R = weights.get("lambda_R", 1.0 / 6.0)
    weight_C = weights.get("lambda_C", 1.0 / 6.0)
    weight_F = weights.get("lambda_F", 1.0 / 6.0)
    weight_O = weights.get("lambda_O", 1.0 / 6.0)
    weight_S = weights.get("lambda_S", 1.0 / 6.0)

    if is_constant_effect:
        extra_weight = weight_S / 5.0
        weight_B += extra_weight
        weight_R += extra_weight
        weight_C += extra_weight
        weight_F += extra_weight
        weight_O += extra_weight
        weight_S = 0.0

    active_components = [scaled_B, scaled_R, scaled_C, scaled_F, scaled_O]

    if not is_constant_effect:
        active_components.append(scaled_S)

    if any(not np.isfinite(x) for x in active_components):
        return np.nan

    loss = (
        weight_B * scaled_B
        + weight_R * scaled_R
        + weight_C * scaled_C
        + weight_F * scaled_F
        + weight_O * scaled_O
        + weight_S * scaled_S
    )

    return float(loss)


def diagnostic_loss_from_metrics(
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
    weights: Dict[str, float],
    is_constant_effect: bool = False,
) -> float:
    """
    Compute diagnostic loss from a metrics dictionary.

    Parameters
    ----------
    metrics : Dict[str, float]
        Metrics dictionary.
    thresholds : Dict[str, float]
        Diagnostic thresholds.
    weights : Dict[str, float]
        Diagnostic weights.
    is_constant_effect : bool, optional
        Whether treatment effects are constant.

    Returns
    -------
    float
        Diagnostic loss.
    """
    components = DiagnosticComponents(
        bias=metrics.get("bias", np.nan),
        rmse=metrics.get("rmse", np.nan),
        coverage_error=metrics.get("coverage_error", np.nan),
        fragility=metrics.get("fragility", np.nan),
        overlap_degradation=metrics.get("overlap_degradation", np.nan),
        estimand_drift=metrics.get("estimand_drift", np.nan),
    )

    return compute_diagnostic_loss(
        components=components,
        thresholds=thresholds,
        weights=weights,
        is_constant_effect=is_constant_effect,
    )


# ============================================================
# 6. Aggregation and Summary Functions
# ============================================================

def aggregate_replication_results(
    replication_results: List[Dict[str, Any]],
    config: SimulationConfig,
) -> Dict[str, Any]:
    """
    Aggregate replication-level results.

    This function is useful when aggregation is performed directly from
    replication dictionaries. Some versions of the package aggregate inside
    simulation.py; therefore the boundary-extraction function below also
    applies the final validity-aware tolerability rule.

    Parameters
    ----------
    replication_results : List[Dict[str, Any]]
        List of replication results.
    config : SimulationConfig
        Simulation configuration.

    Returns
    -------
    Dict[str, Any]
        Aggregated diagnostics.
    """
    if len(replication_results) == 0:
        return {}

    tau_hats = []
    targets = []
    errors = []
    squared_errors = []
    coverages = []
    fragilities = []
    overlaps = []
    drifts = []
    valid_flags = []

    for result in replication_results:
        tau_hat = result.get("tau_hat", np.nan)
        target = result.get("target", np.nan)
        valid = result.get("valid", False)

        if valid and np.isfinite(tau_hat) and np.isfinite(target):
            tau_hats.append(tau_hat)
            targets.append(target)
            errors.append(tau_hat - target)
            squared_errors.append((tau_hat - target) ** 2)
            valid_flags.append(True)
        else:
            valid_flags.append(False)

        coverage = result.get("coverage", np.nan)
        if np.isfinite(coverage):
            coverages.append(coverage)

        fragility = result.get("fragility", np.nan)
        if np.isfinite(fragility):
            fragilities.append(fragility)

        overlap = result.get("overlap_degradation", np.nan)
        if np.isfinite(overlap):
            overlaps.append(overlap)

        drift = result.get("estimand_drift", np.nan)
        if np.isfinite(drift):
            drifts.append(drift)

    n_valid = len(tau_hats)
    n_total = len(replication_results)
    valid_rate = n_valid / n_total if n_total > 0 else 0.0

    if n_valid > 0:
        bias = np.nanmean(errors)
        rmse = np.sqrt(np.nanmean(squared_errors))
        coverage = np.nanmean(coverages) if coverages else np.nan
        fragility = np.nanmean(fragilities) if fragilities else np.nan
        overlap_degradation = np.nanmean(overlaps) if overlaps else np.nan
        estimand_drift = np.nanmean(drifts) if drifts else np.nan
    else:
        bias = np.nan
        rmse = np.nan
        coverage = np.nan
        fragility = np.nan
        overlap_degradation = np.nan
        estimand_drift = np.nan

    nominal_coverage = 1.0 - config.alpha
    coverage_error_value = (
        max(0.0, nominal_coverage - coverage)
        if np.isfinite(coverage)
        else np.nan
    )

    thresholds = config.get_diagnostic_thresholds()
    weights = config.get_loss_weights()

    is_constant = False
    if replication_results and "effect_type" in replication_results[0]:
        is_constant = replication_results[0].get("effect_type") == "constant"

    loss = diagnostic_loss_from_metrics(
        metrics={
            "bias": bias,
            "rmse": rmse,
            "coverage_error": coverage_error_value,
            "fragility": fragility,
            "overlap_degradation": overlap_degradation,
            "estimand_drift": estimand_drift,
        },
        thresholds=thresholds,
        weights=weights,
        is_constant_effect=is_constant,
    )

    tolerable_loss_only = bool(np.isfinite(loss) and loss <= 1.0)
    tolerable = bool(tolerable_loss_only and valid_rate >= 0.90)

    return {
        "bias": bias,
        "rmse": rmse,
        "coverage": coverage,
        "coverage_error": coverage_error_value,
        "fragility": fragility,
        "overlap_degradation": overlap_degradation,
        "estimand_drift": estimand_drift,
        "valid_rate": valid_rate,
        "loss": loss,
        "tolerable_loss_only": tolerable_loss_only,
        "tolerable": tolerable,
        "n_valid": n_valid,
        "n_total": n_total,
        "is_constant_effect": is_constant,
    }


def extract_tolerance_boundary(
    summary_df: pd.DataFrame,
    loss_col: str = "loss",
    delta_col: str = "delta",
    group_cols: List[str] = None,
    validity_col: str = "valid_rate",
    validity_threshold: float = 0.90,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract tolerance boundaries from summary results.

    Two boundary types are reported:
    1. Simple boundary:
       max delta where final tolerability is satisfied.
    2. Conservative continuous boundary:
       max delta such that final tolerability is satisfied for all lower deltas.

    Final tolerability is validity-aware:
       tolerable = (loss <= 1.0) and (valid_rate >= validity_threshold)

    If valid_rate is not present, the function falls back to loss-only
    tolerability for backward compatibility.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary results.
    loss_col : str, optional
        Name of the diagnostic-loss column.
    delta_col : str, optional
        Name of the delta column.
    group_cols : List[str], optional
        Scenario and method grouping columns.
    validity_col : str, optional
        Name of the validity-rate column.
    validity_threshold : float, optional
        Minimum acceptable validity rate.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Boundaries dataframe and relative-ratios dataframe.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if loss_col not in summary_df.columns:
        raise ValueError(f"Column '{loss_col}' not found in summary_df.")

    if delta_col not in summary_df.columns:
        raise ValueError(f"Column '{delta_col}' not found in summary_df.")

    if group_cols is None:
        possible_group_cols = [
            "scenario_id",
            "q_regime",
            "outcome_model",
            "effect_type",
            "method",
            "n",
            "rho",
            "pi_treat",
            "s_a",
            "s_b",
        ]
        group_cols = [col for col in possible_group_cols if col in summary_df.columns]

    missing_group_cols = [col for col in group_cols if col not in summary_df.columns]
    if missing_group_cols:
        raise ValueError(f"Grouping columns not found in summary_df: {missing_group_cols}")

    if "method" not in group_cols and "method" in summary_df.columns:
        group_cols = group_cols + ["method"]

    boundary_rows = []

    for keys, g in summary_df.groupby(group_cols, dropna=False):
        g_sorted = g.sort_values(delta_col).copy()

        # Keep loss-only decision separately.
        g_sorted["tolerable_loss_only"] = g_sorted[loss_col] <= 1.0

        # Final validity-aware tolerability.
        if validity_col in g_sorted.columns:
            g_sorted["tolerable"] = (
                g_sorted["tolerable_loss_only"]
                & (g_sorted[validity_col] >= validity_threshold)
            )
        elif "tolerable" in g_sorted.columns:
            # Backward-compatible fallback if a final tolerability column
            # already exists but valid_rate is absent.
            g_sorted["tolerable"] = g_sorted["tolerable"].astype(bool)
        else:
            g_sorted["tolerable"] = g_sorted["tolerable_loss_only"]

        continuous_ok = []
        still_ok = True

        for ok in g_sorted["tolerable"].tolist():
            still_ok = bool(still_ok and ok)
            continuous_ok.append(still_ok)

        g_sorted["continuous_tolerable"] = continuous_ok

        if np.any(g_sorted["tolerable"]):
            delta_star = float(g_sorted.loc[g_sorted["tolerable"], delta_col].max())
        else:
            delta_star = 0.0

        if np.any(g_sorted["continuous_tolerable"]):
            delta_star_conservative = float(
                g_sorted.loc[g_sorted["continuous_tolerable"], delta_col].max()
            )
        else:
            delta_star_conservative = 0.0

        delta0_mask = g_sorted[delta_col] == 0.0

        if np.any(delta0_mask):
            loss_at_delta0 = float(g_sorted.loc[delta0_mask, loss_col].iloc[0])
            tolerable_at_delta0 = bool(g_sorted.loc[delta0_mask, "tolerable"].iloc[0])
            tolerable_loss_only_at_delta0 = bool(
                g_sorted.loc[delta0_mask, "tolerable_loss_only"].iloc[0]
            )

            if validity_col in g_sorted.columns:
                valid_rate_at_delta0 = float(g_sorted.loc[delta0_mask, validity_col].iloc[0])
            else:
                valid_rate_at_delta0 = np.nan
        else:
            loss_at_delta0 = np.nan
            tolerable_at_delta0 = False
            tolerable_loss_only_at_delta0 = False
            valid_rate_at_delta0 = np.nan

        if validity_col in g_sorted.columns:
            min_valid_rate = float(g_sorted[validity_col].min())
        else:
            min_valid_rate = np.nan

        row = {}

        if isinstance(keys, tuple):
            for i, col in enumerate(group_cols):
                row[col] = keys[i]
        else:
            row[group_cols[0]] = keys

        row.update(
            {
                "delta_star": delta_star,
                "delta_star_conservative": delta_star_conservative,
                "loss_at_delta0": loss_at_delta0,
                "tolerable_at_delta0": tolerable_at_delta0,
                "tolerable_loss_only_at_delta0": tolerable_loss_only_at_delta0,
                "valid_rate_at_delta0": valid_rate_at_delta0,
                "min_valid_rate": min_valid_rate,
                "ever_fragile": bool(np.any(~g_sorted["tolerable"])),
            }
        )

        boundary_rows.append(row)

    boundaries_df = pd.DataFrame(boundary_rows)

    if not boundaries_df.empty and "method" in boundaries_df.columns:
        rank_cols = [col for col in group_cols if col != "method"]

        boundaries_df["rank"] = boundaries_df.groupby(rank_cols)[
            "delta_star_conservative"
        ].rank(method="min", ascending=False)

    if "method" in boundaries_df.columns and "PSM" in boundaries_df["method"].values:
        ratios_df = compute_relative_ratios(boundaries_df, "PSM")
    else:
        ratios_df = pd.DataFrame()

    return boundaries_df, ratios_df


def compute_relative_ratios(
    boundaries_df: pd.DataFrame,
    reference_method: str = "PSM",
    boundary_col: str = "delta_star_conservative",
) -> pd.DataFrame:
    """
    Compute relative tolerance ratios for a reference method against all
    other methods within the same scenario.

    Ratio definition:
        relative_tolerance = reference_method_delta_star / other_method_delta_star

    Interpretation:
        ratio < 1: reference method is less tolerant than the comparator.
        ratio = 1: equal tolerance.
        ratio > 1: reference method is more tolerant than the comparator.

    Parameters
    ----------
    boundaries_df : pd.DataFrame
        Tolerance-boundary dataframe.
    reference_method : str, optional
        Method used as the numerator.
    boundary_col : str, optional
        Boundary column to compare.

    Returns
    -------
    pd.DataFrame
        Relative-tolerance ratios.
    """
    if boundaries_df is None or boundaries_df.empty:
        return pd.DataFrame()

    if "method" not in boundaries_df.columns:
        return pd.DataFrame()

    if boundary_col not in boundaries_df.columns:
        return pd.DataFrame()

    if reference_method not in boundaries_df["method"].values:
        return pd.DataFrame()

    exclude_cols = {
        "method",
        "delta_star",
        "delta_star_conservative",
        "loss_at_delta0",
        "tolerable_at_delta0",
        "tolerable_loss_only_at_delta0",
        "valid_rate_at_delta0",
        "min_valid_rate",
        "ever_fragile",
        "rank",
    }

    scenario_cols = [col for col in boundaries_df.columns if col not in exclude_cols]

    ratio_rows = []

    for keys, g in boundaries_df.groupby(scenario_cols, dropna=False):
        ref_row = g[g["method"] == reference_method]

        if ref_row.empty:
            continue

        ref_delta = float(ref_row[boundary_col].iloc[0])

        if not isinstance(keys, tuple):
            keys = (keys,)

        scenario_data = dict(zip(scenario_cols, keys))

        for _, row in g.iterrows():
            method = row["method"]

            if method == reference_method:
                continue

            method_delta = float(row[boundary_col])

            if method_delta == 0.0:
                relative_tolerance = np.nan if ref_delta > 0 else 1.0
            else:
                relative_tolerance = ref_delta / method_delta

            ratio_row = scenario_data.copy()
            ratio_row.update(
                {
                    "comparison": f"{reference_method}:{method}",
                    "ref_delta_star": ref_delta,
                    "method_delta_star": method_delta,
                    "relative_tolerance": relative_tolerance,
                }
            )

            ratio_rows.append(ratio_row)

    return pd.DataFrame(ratio_rows)


# ============================================================
# 7. Replication-Level Diagnostics
# ============================================================

def compute_replication_diagnostics(
    data: Dict[str, np.ndarray],
    estimator_result: Dict[str, Any],
    config: SimulationConfig,
) -> Dict[str, Any]:
    """
    Compute diagnostics for a single replication.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Generated data.
    estimator_result : Dict[str, Any]
        Estimator result.
    config : SimulationConfig
        Simulation configuration.

    Returns
    -------
    Dict[str, Any]
        Replication diagnostics.
    """
    X = data["X"]
    T = data["T"]
    tau_x = data["tau_x"]
    e0 = data["e0"]
    ehat = estimator_result.get("ehat", None)

    treated_weights = estimator_result.get("treated_weights", np.zeros(len(T)))
    control_weights = estimator_result.get("control_weights", np.zeros(len(T)))

    tau_hat = estimator_result.get("tau_hat", np.nan)
    target = estimator_result.get("target", np.nan)
    method = estimator_result.get("method", "unknown")

    fragility = design_fragility(
        X=X,
        treated_weights=treated_weights,
        control_weights=control_weights,
        extended=True,
    )

    overlap = effective_support_degradation(
        e0=e0,
        treated_weights=treated_weights,
        control_weights=control_weights,
        eta=config.eta_overlap,
        kappa=config.kappa_weight,
    )

    score_overlap = (
        overlap_degradation(e0=e0, ehat=ehat, eta=config.eta_overlap)
        if ehat is not None
        else np.nan
    )

    if tau_x is not None:
        drift = estimand_drift(
            treated_weights=treated_weights,
            tau_x=tau_x,
            T=T,
        )
    else:
        drift = np.nan

    return {
        "method": method,
        "tau_hat": tau_hat,
        "target": target,
        "fragility": fragility,
        "overlap_degradation": overlap,
        "score_overlap_degradation": score_overlap,
        "estimand_drift": drift,
        "valid": estimator_result.get("valid", False),
    }