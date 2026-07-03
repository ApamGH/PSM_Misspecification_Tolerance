"""
Simulation Runner
=================

Main simulation engine that orchestrates the data generation, estimation,
and diagnostic evaluation across scenarios and replications.

Implements the simulation design from Section 5 of the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import time

from .config import SimulationConfig
from .data_generation import DataGenerator, fit_propensity, fit_outcome_control
from .estimators.factory import get_estimator
from .diagnostics import (
    compute_replication_diagnostics,
    aggregate_replication_results,
    extract_tolerance_boundary,
    design_fragility,
    overlap_degradation,
    estimand_drift,
    effective_support_degradation,
)
from .utils import (
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    ProgressTracker,
    warn_and_continue,
)


class SimulationRunner:
    """
    Main simulation runner.
    
    Orchestrates the complete simulation:
    1. Generates scenarios from configuration
    2. Runs Monte Carlo replications
    3. Aggregates results
    4. Extracts tolerance boundaries
    5. Saves outputs
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    logger : logging.Logger, optional
        Logger instance
    """
    
    def __init__(self, config: SimulationConfig, logger=None):
        self.config = config
        self.logger = logger or setup_logger("SimulationRunner")
        
        # Random number generator
        self.rng = np.random.default_rng(config.random_seed)
        
        # Data generator
        self.data_generator = DataGenerator(config, self.rng)
        
        # Results storage
        self.replication_results = []
        self.summary_results = None
        self.boundaries = None
        self.ratios = None
        
        # Progress tracking
        self.total_scenarios = 0
        self.completed_scenarios = 0
    
    def run(self, resume_from: Optional[str] = None) -> pd.DataFrame:
        """
        Run the full simulation.
        
        Parameters
        ----------
        resume_from : str, optional
            Path to checkpoint file to resume from
        
        Returns
        -------
        pd.DataFrame
            Replication-level results
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Misspecification-Tolerance Simulation")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration: {self.config.__dict__}")
        
        # Resume from checkpoint if provided
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Generate scenarios
        scenarios = self._generate_scenarios()
        self.total_scenarios = len(scenarios)
        
        self.logger.info(f"Total scenarios: {self.total_scenarios}")
        
        # Run scenarios
        start_time = time.time()
        
        for idx, scenario in enumerate(scenarios):
            self.logger.info(f"Scenario {idx+1}/{self.total_scenarios}: {scenario}")
            
            # Skip if already completed
            if self._scenario_completed(scenario):
                self.logger.info(f"  Skipping (already completed)")
                continue
            
            # Run scenario
            try:
                scenario_results = self._run_scenario(scenario)
                self.replication_results.extend(scenario_results)
                
                # Save checkpoint
                if self.config.save_checkpoints and (idx + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                self.completed_scenarios += 1
                self.logger.info(f"  Completed: {len(scenario_results)} replications")
                
            except Exception as e:
                self.logger.error(f"  Scenario failed: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Completed {self.completed_scenarios}/{self.total_scenarios} scenarios")
        
        # Aggregate results
        self._aggregate_results()
        
        # Extract boundaries
        self._extract_boundaries()
        
        # Convert to DataFrame
        return pd.DataFrame(self.replication_results)
    
    def _generate_scenarios(self) -> List[Dict[str, Any]]:
        """
        Generate all scenarios from the configuration.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of scenario dictionaries
        """
        scenarios = []
        
        # Use the full factorial design from config
        n_values = self.config.n_values
        rho_values = self.config.rho_values
        pi_treat_values = self.config.pi_treat_values
        s_a_values = self.config.s_a_values
        s_b_values = self.config.s_b_values
        
        for n in n_values:
            for rho in rho_values:
                for pi_treat in pi_treat_values:
                    for s_a in s_a_values:
                        for s_b in s_b_values:
                            for q_regime in self.config.q_regimes:
                                for outcome_model in self.config.outcome_models:
                                    for effect_type in self.config.treatment_effects:
                                        # For each combination, include all deltas
                                        scenario = {
                                            'n': n,
                                            'rho': rho,
                                            'pi_treat': pi_treat,
                                            's_a': s_a,
                                            's_b': s_b,
                                            'q_regime': q_regime,
                                            'outcome_model': outcome_model,
                                            'effect_type': effect_type,
                                        }
                                        scenarios.append(scenario)
        
        return scenarios
    
    def _scenario_completed(self, scenario: Dict[str, Any]) -> bool:
        """
        Check if a scenario has already been completed.
        
        Parameters
        ----------
        scenario : Dict[str, Any]
            Scenario dictionary
        
        Returns
        -------
        bool
            True if scenario is completed
        """
        # Simple check: look for any results with same scenario
        for result in self.replication_results:
            match = True
            for key, value in scenario.items():
                if result.get(key) != value:
                    match = False
                    break
            if match:
                return True
        return False
    
    def _run_scenario(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run one scenario with all deltas and replications.
        
        Parameters
        ----------
        scenario : Dict[str, Any]
            Scenario parameters
        
        Returns
        -------
        List[Dict[str, Any]]
            List of replication results
        """
        results = []
        
        # Extract scenario parameters
        n = scenario['n']
        rho = scenario['rho']
        pi_treat = scenario['pi_treat']
        s_a = scenario['s_a']
        s_b = scenario['s_b']
        q_regime = scenario['q_regime']
        outcome_model = scenario['outcome_model']
        effect_type = scenario['effect_type']
        
        # Use the data generator's calibration data for this scenario
        # (already generated with the base rho)
        
        # For each delta
        for delta in self.config.delta_grid:
            # Generate all replications for this delta
            delta_results = self._run_delta(
                delta=delta,
                n=n,
                rho=rho,
                pi_treat=pi_treat,
                s_a=s_a,
                s_b=s_b,
                q_regime=q_regime,
                outcome_model=outcome_model,
                effect_type=effect_type,
            )
            results.extend(delta_results)
        
        return results
    
    def _run_delta(
        self,
        delta: float,
        n: int,
        rho: float,
        pi_treat: float,
        s_a: float,
        s_b: float,
        q_regime: str,
        outcome_model: str,
        effect_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Run all replications for one delta value.
        
        Parameters
        ----------
        delta : float
            Misspecification intensity
        n : int
            Sample size
        rho : float
            Covariate correlation
        pi_treat : float
            Treatment prevalence
        s_a : float
            Overlap severity
        s_b : float
            Prognostic strength
        q_regime : str
            Misspecification regime
        outcome_model : str
            Outcome model type
        effect_type : str
            Treatment effect type
        
        Returns
        -------
        List[Dict[str, Any]]
            List of replication results
        """
        R = self.config.R
        
        # Use parallel processing if requested
        if self.config.n_jobs != 1:
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._run_replication)(
                    delta=delta,
                    n=n,
                    rho=rho,
                    pi_treat=pi_treat,
                    s_a=s_a,
                    s_b=s_b,
                    q_regime=q_regime,
                    outcome_model=outcome_model,
                    effect_type=effect_type,
                    replication_idx=r,
                )
                for r in range(R)
            )
        else:
            results = []
            for r in range(R):
                result = self._run_replication(
                    delta=delta,
                    n=n,
                    rho=rho,
                    pi_treat=pi_treat,
                    s_a=s_a,
                    s_b=s_b,
                    q_regime=q_regime,
                    outcome_model=outcome_model,
                    effect_type=effect_type,
                    replication_idx=r,
                )
                if result is not None:
                    results.append(result)
        
        # Filter out failed replications and flatten method-level rows.
        flat_results = []
        for item in results:
            if item is None:
                continue
            if isinstance(item, list):
                flat_results.extend(item)
            else:
                flat_results.append(item)
        return flat_results
    
    def _run_replication(
        self,
        delta: float,
        n: int,
        rho: float,
        pi_treat: float,
        s_a: float,
        s_b: float,
        q_regime: str,
        outcome_model: str,
        effect_type: str,
        replication_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Run one Monte Carlo replication.
        
        Steps:
        1. Generate data
        2. Fit propensity score
        3. Fit outcome models (for AIPW)
        4. Run all estimators
        5. Compute diagnostics
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Replication results, or None if failed
        """
        try:
            # 1. Generate data
            data = self.data_generator.generate_dataset(
                delta=delta,
                q_regime=q_regime,
                outcome_model=outcome_model,
                effect_type=effect_type,
                n=n,
                rho=rho,
                pi_treat=pi_treat,
                s_a=s_a,
                s_b=s_b,
            )
            
            X = data['X']
            T = data['T']
            Y = data['Y']
            tau_x = data['tau_x']
            e0 = data['e0']
            
            # 2. Fit working propensity score
            ehat = fit_propensity(X, T, q_regime)
            
            # 3. Fit outcome models for AIPW
            # AIPW_correct: extended when outcome model is nonlinear
            extended_correct = outcome_model == "nonlinear"
            m0hat_correct = fit_outcome_control(X, T, Y, extended=extended_correct)
            
            # AIPW_restricted: always linear
            m0hat_restricted = fit_outcome_control(X, T, Y, extended=False)
            
            # 4. Run all estimators
            results = []
            
            for method_name in self.config.methods:
                # Get estimator
                estimator = get_estimator(method_name, self.config)
                
                # Special handling for AIPW estimators
                if method_name == "AIPW_correct":
                    est_result = estimator.estimate(
                        X=X, T=T, Y=Y, ehat=ehat, m0hat=m0hat_correct, tau_x=tau_x, e0=e0
                    )
                elif method_name == "AIPW_restricted":
                    est_result = estimator.estimate(
                        X=X, T=T, Y=Y, ehat=ehat, m0hat=m0hat_restricted, tau_x=tau_x, e0=e0
                    )
                else:
                    est_result = estimator.estimate(
                        X=X, T=T, Y=Y, ehat=ehat, tau_x=tau_x, e0=e0
                    )
                
                # Determine target
                if method_name == "OW":
                    target = data['ato_sample']
                else:
                    target = data['att_sample']
                
                # Get confidence interval (from metadata)
                lower = est_result.metadata.get('lower', np.nan)
                upper = est_result.metadata.get('upper', np.nan)
                
                # Coverage
                if np.isfinite(lower) and np.isfinite(upper):
                    coverage = 1.0 if lower <= target <= upper else 0.0
                else:
                    coverage = np.nan
                
                # Compute diagnostics
                diagnostics = compute_replication_diagnostics(
                    data=data,
                    estimator_result={
                        'method': method_name,
                        'tau_hat': est_result.tau_hat,
                        'target': target,
                        'treated_weights': est_result.treated_weights,
                        'control_weights': est_result.control_weights,
                        'ehat': ehat,
                        'valid': est_result.valid,
                    },
                    config=self.config
                )
                
                # Build result dictionary
                result = {
                    # Scenario identifiers
                    'scenario_id': f"{q_regime}_{outcome_model}_{effect_type}",
                    'q_regime': q_regime,
                    'outcome_model': outcome_model,
                    'effect_type': effect_type,
                    'delta': delta,
                    'replication': replication_idx,
                    'n': n,
                    'rho': rho,
                    'pi_treat': pi_treat,
                    's_a': s_a,
                    's_b': s_b,
                    
                    # Method
                    'method': method_name,
                    
                    # Estimates
                    'tau_hat': est_result.tau_hat,
                    'target': target,
                    'error': est_result.tau_hat - target if est_result.valid else np.nan,
                    'squared_error': (est_result.tau_hat - target) ** 2 if est_result.valid else np.nan,
                    
                    # Diagnostics
                    'fragility': diagnostics['fragility'],
                    'overlap_degradation': diagnostics['overlap_degradation'],
                    'score_overlap_degradation': diagnostics.get('score_overlap_degradation', np.nan),
                    'estimand_drift': diagnostics['estimand_drift'],
                    'coverage': coverage,
                    
                    # Validity
                    'valid': est_result.valid,
                    
                    # Additional metadata
                    'sigma_y': data['sigma_y'],
                    'sigma_tau_treated': data['sigma_tau_treated'],
                    'n_treated': int(np.sum(T)),
                    'n_control': int(n - np.sum(T)),
                    'n_retained': est_result.metadata.get('n_retained', n),
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.warning(
                f"Replication failed: delta={delta}, replication={replication_idx}, "
                f"q_regime={q_regime}, error={e}"
            )
            return None
    
    def _aggregate_results(self) -> None:
        """Aggregate replication results into summary.

        The loss uses baseline-adjusted effective-support degradation:
            overlap_relative = max(0, overlap_raw(delta) - overlap_raw(delta=0)).

        Raw overlap/support values are retained for descriptive reporting.

        Final tolerability is validity-aware:
            tolerable = (loss <= 1.0) and (valid_rate >= 0.90)

        The loss-only decision is also retained as:
            tolerable_loss_only
        """
        if not self.replication_results:
            self.logger.warning("No results to aggregate")
            return

        self.logger.info("Aggregating results...")
        df = pd.DataFrame(self.replication_results)

        group_cols = [
            'scenario_id', 'q_regime', 'outcome_model', 'effect_type',
            'delta', 'method', 'n', 'rho', 'pi_treat', 's_a', 's_b'
        ]

        grouped = df.groupby(group_cols, dropna=False)
        summary_list = []

        for keys, g in grouped:
            key_dict = dict(zip(group_cols, keys)) if isinstance(keys, tuple) else {group_cols[0]: keys}

            valid_mask = g['valid'] == True
            valid_g = g[valid_mask]
            valid_rate = float(valid_mask.mean()) if len(g) > 0 else 0.0

            if len(valid_g) == 0:
                summary_list.append({
                    **key_dict,
                    'bias': np.nan,
                    'rmse': np.nan,
                    'coverage': np.nan,
                    'coverage_error': np.nan,
                    'fragility': np.nan,
                    'overlap_raw': np.nan,
                    'score_overlap_degradation': np.nan,
                    'estimand_drift': np.nan,
                    'sigma_y': np.nan,
                    'sigma_tau_treated': np.nan,
                    'valid_rate': 0.0,
                })
                continue

            coverage = valid_g['coverage'].mean() if 'coverage' in valid_g else np.nan
            nominal_coverage = 1.0 - self.config.alpha
            coverage_error = max(0.0, nominal_coverage - coverage) if np.isfinite(coverage) else np.nan

            summary_list.append({
                **key_dict,
                'bias': valid_g['error'].mean(),
                'rmse': float(np.sqrt(valid_g['squared_error'].mean())),
                'coverage': coverage,
                'coverage_error': coverage_error,
                'fragility': valid_g['fragility'].mean(),
                'overlap_raw': valid_g['overlap_degradation'].mean(),
                'score_overlap_degradation': (
                    valid_g['score_overlap_degradation'].mean()
                    if 'score_overlap_degradation' in valid_g
                    else np.nan
                ),
                'estimand_drift': valid_g['estimand_drift'].mean(),
                'sigma_y': valid_g['sigma_y'].mean(),
                'sigma_tau_treated': valid_g['sigma_tau_treated'].mean(),
                'valid_rate': valid_rate,
            })

        summary = pd.DataFrame(summary_list)

        # Baseline-adjusted overlap: compare each method to its own delta=0 support.
        baseline_cols = [
            'scenario_id', 'q_regime', 'outcome_model', 'effect_type',
            'method', 'n', 'rho', 'pi_treat', 's_a', 's_b'
        ]

        baseline_overlap = (
            summary.loc[summary['delta'] == 0.0, baseline_cols + ['overlap_raw']]
            .rename(columns={'overlap_raw': 'overlap_delta0'})
        )

        summary = summary.merge(baseline_overlap, on=baseline_cols, how='left')

        summary['overlap_relative'] = (
            summary['overlap_raw'] - summary['overlap_delta0']
        ).clip(lower=0.0)

        # Compute diagnostic loss with row-specific empirical thresholds.
        weights = self.config.get_loss_weights()
        losses = []

        from .diagnostics import diagnostic_loss_from_metrics

        for _, row in summary.iterrows():
            is_constant = row.get('effect_type') == 'constant'

            eps_S = self.config.eps_drift_multiplier * row.get('sigma_tau_treated', np.nan)

            if is_constant or not np.isfinite(eps_S) or eps_S <= 0:
                eps_S = np.nan

            thresholds = {
                'eps_B': self.config.eps_bias_multiplier * row.get('sigma_y', np.nan),
                'eps_R': self.config.eps_rmse_multiplier * row.get('sigma_y', np.nan),
                'eps_C': self.config.eps_coverage,
                'eps_F': self.config.eps_fragility,
                'eps_O': self.config.eps_overlap,
                'eps_S': eps_S,
            }

            loss = diagnostic_loss_from_metrics(
                metrics={
                    'bias': row.get('bias', np.nan),
                    'rmse': row.get('rmse', np.nan),
                    'coverage_error': row.get('coverage_error', np.nan),
                    'fragility': row.get('fragility', np.nan),
                    'overlap_degradation': row.get('overlap_relative', np.nan),
                    'estimand_drift': row.get('estimand_drift', np.nan),
                },
                thresholds=thresholds,
                weights=weights,
                is_constant_effect=is_constant,
            )

            losses.append(loss)

        summary['loss'] = losses

        # Loss-only tolerability.
        # This preserves the original diagnostic-loss rule for checking.
        summary['tolerable_loss_only'] = summary['loss'] <= 1.0

        # Final tolerability rule.
        # A method is tolerable only if diagnostic loss is acceptable
        # and valid estimates are obtained in at least 90% of replications.
        summary['tolerable'] = (
            summary['tolerable_loss_only']
            & (summary['valid_rate'] >= 0.90)
        )

        self.summary_results = summary
        self.logger.info(f"Aggregated {len(self.summary_results)} summary rows")


    def _extract_boundaries(self) -> None:
        """Extract tolerance boundaries from summary results."""
        if self.summary_results is None:
            self.logger.warning("No summary results to extract boundaries from")
            return
        
        self.logger.info("Extracting tolerance boundaries...")
        
        from .diagnostics import extract_tolerance_boundary
        
        # Group only by scenario identifiers and method. Do not include
        # diagnostic outcomes such as bias, RMSE, or overlap in the grouping.
        group_cols = [
            'scenario_id', 'q_regime', 'outcome_model', 'effect_type',
            'method', 'n', 'rho', 'pi_treat', 's_a', 's_b'
        ]
        
        self.boundaries, self.ratios = extract_tolerance_boundary(
            self.summary_results,
            group_cols=group_cols
        )
        
        self.logger.info(f"Extracted boundaries for {len(self.boundaries)} scenarios")
        self.logger.info(f"Computed ratios for {len(self.ratios)} comparisons")
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint for resuming."""
        checkpoint_path = Path(self.config.output_dir) / "checkpoint.pkl"
        
        checkpoint = {
            'config': self.config,
            'replication_results': self.replication_results,
            'completed_scenarios': self.completed_scenarios,
            'timestamp': time.time(),
        }
        
        save_checkpoint(checkpoint, str(checkpoint_path))
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        checkpoint = load_checkpoint(path)
        
        self.replication_results = checkpoint.get('replication_results', [])
        self.completed_scenarios = checkpoint.get('completed_scenarios', 0)
        
        self.logger.info(f"Loaded checkpoint from {path}")
        self.logger.info(f"  Completed scenarios: {self.completed_scenarios}")
        self.logger.info(f"  Existing replications: {len(self.replication_results)}")
    
    def save_results(self, output_path: Optional[str] = None) -> None:
        """
        Save all results to disk.
        
        Parameters
        ----------
        output_path : str, optional
            Output file path
        """
        if output_path is None:
            output_path = Path(self.config.output_dir) / "results.pkl"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'config': self.config.__dict__,
            'replication_results': self.replication_results,
            'summary_results': self.summary_results,
            'boundaries': self.boundaries,
            'ratios': self.ratios,
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def export_to_csv(self, output_dir: Optional[str] = None) -> None:
        """
        Export results to CSV files.
        
        Parameters
        ----------
        output_dir : str, optional
            Output directory
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Replication results
        if self.replication_results:
            df = pd.DataFrame(self.replication_results)
            df.to_csv(output_dir / "replication_results.csv", index=False)
            self.logger.info(f"Replication results exported to {output_dir}/replication_results.csv")
        
        # Summary results
        if self.summary_results is not None:
            self.summary_results.to_csv(output_dir / "summary_results.csv", index=False)
            self.logger.info(f"Summary results exported to {output_dir}/summary_results.csv")
        
        # Boundaries
        if self.boundaries is not None:
            self.boundaries.to_csv(output_dir / "tolerance_boundaries.csv", index=False)
            self.logger.info(f"Boundaries exported to {output_dir}/tolerance_boundaries.csv")
        
        # Ratios
        if self.ratios is not None and not self.ratios.empty:
            self.ratios.to_csv(output_dir / "relative_tolerance_ratios.csv", index=False)
            self.logger.info(f"Ratios exported to {output_dir}/relative_tolerance_ratios.csv")