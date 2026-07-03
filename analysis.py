"""
Analysis Module
===============

Functions for analyzing simulation results after they've been generated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle


def load_results(path: str) -> Dict[str, Any]:
    """
    Load saved simulation results.
    
    Parameters
    ----------
    path : str
        Path to results file (.pkl)
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_method_ranks(
    boundaries_df: pd.DataFrame,
    method_col: str = 'method',
    boundary_col: str = 'delta_star_conservative',
    scenario_cols: List[str] = None
) -> pd.DataFrame:
    """
    Rank methods by tolerance boundary within each scenario.
    
    Parameters
    ----------
    boundaries_df : pd.DataFrame
        Boundaries dataframe
    method_col : str
        Column name for method
    boundary_col : str
        Column name for boundary value
    scenario_cols : List[str], optional
        Columns that define a scenario
    
    Returns
    -------
    pd.DataFrame
        Boundaries with ranks added
    """
    if scenario_cols is None:
        scenario_cols = [col for col in boundaries_df.columns 
                        if col not in [method_col, boundary_col]]
    
    df = boundaries_df.copy()
    
    df['rank'] = df.groupby(scenario_cols)[boundary_col].rank(
        method='min',
        ascending=False
    )
    
    return df


def compute_best_worst_methods(
    boundaries_df: pd.DataFrame,
    method_col: str = 'method',
    boundary_col: str = 'delta_star_conservative',
    scenario_cols: List[str] = None
) -> pd.DataFrame:
    """
    Find best and worst methods for each scenario.
    
    Parameters
    ----------
    boundaries_df : pd.DataFrame
        Boundaries dataframe
    method_col : str
        Column name for method
    boundary_col : str
        Column name for boundary value
    scenario_cols : List[str], optional
        Columns that define a scenario
    
    Returns
    -------
    pd.DataFrame
        Best and worst methods per scenario
    """
    if scenario_cols is None:
        scenario_cols = [col for col in boundaries_df.columns 
                        if col not in [method_col, boundary_col]]
    
    results = []
    
    for keys, g in boundaries_df.groupby(scenario_cols, dropna=False):
        # Best (highest boundary)
        best_row = g.loc[g[boundary_col].idxmax()]
        worst_row = g.loc[g[boundary_col].idxmin()]
        
        row = {}
        if isinstance(keys, tuple):
            for i, col in enumerate(scenario_cols):
                row[col] = keys[i]
        else:
            row[scenario_cols[0]] = keys
        
        row.update({
            'best_method': best_row[method_col],
            'best_boundary': best_row[boundary_col],
            'worst_method': worst_row[method_col],
            'worst_boundary': worst_row[boundary_col],
            'boundary_range': best_row[boundary_col] - worst_row[boundary_col],
        })
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_summary_statistics(
    summary_df: pd.DataFrame,
    method_col: str = 'method',
    metric_cols: List[str] = None
) -> pd.DataFrame:
    """
    Compute summary statistics across scenarios for each method.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary results
    method_col : str
        Column name for method
    metric_cols : List[str], optional
        Columns with metrics to summarize
    
    Returns
    -------
    pd.DataFrame
        Summary statistics per method
    """
    if metric_cols is None:
        metric_cols = ['bias', 'rmse', 'coverage', 'loss']
    
    stats = []
    
    for method in summary_df[method_col].unique():
        method_data = summary_df[summary_df[method_col] == method]
        
        row = {'method': method}
        
        for metric in metric_cols:
            if metric in method_data.columns:
                values = method_data[metric].dropna()
                if len(values) > 0:
                    row[f'{metric}_mean'] = values.mean()
                    row[f'{metric}_std'] = values.std()
                    row[f'{metric}_median'] = values.median()
                    row[f'{metric}_min'] = values.min()
                    row[f'{metric}_max'] = values.max()
        
        stats.append(row)
    
    return pd.DataFrame(stats)


def identify_breakpoints(
    summary_df: pd.DataFrame,
    delta_col: str = 'delta',
    loss_col: str = 'loss',
    method_col: str = 'method',
    threshold: float = 1.0,
    group_cols: List[str] = None
) -> pd.DataFrame:
    """
    Identify delta values where methods cross the tolerance threshold.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary results
    delta_col : str
        Column name for delta
    loss_col : str
        Column name for loss
    method_col : str
        Column name for method
    threshold : float
        Loss threshold (default: 1.0)
    group_cols : List[str], optional
        Columns to group by
    
    Returns
    -------
    pd.DataFrame
        Breakpoints per method and scenario
    """
    if group_cols is None:
        group_cols = [col for col in summary_df.columns 
                     if col not in [delta_col, loss_col, method_col]]
    
    breakpoints = []
    
    for keys, g in summary_df.groupby(group_cols + [method_col], dropna=False):
        g_sorted = g.sort_values(delta_col)
        
        # Find where loss > threshold
        crossed = g_sorted[loss_col] > threshold
        
        if np.any(crossed):
            # First delta where crossed
            first_cross = g_sorted[crossed].iloc[0]
            breakpoint = first_cross[delta_col]
        else:
            # Never crosses
            breakpoint = g_sorted[delta_col].max() + 0.1
        
        row = {}
        if isinstance(keys, tuple):
            for i, col in enumerate(group_cols + [method_col]):
                row[col] = keys[i]
        else:
            row[group_cols[0]] = keys
        
        row['breakpoint'] = breakpoint
        row['crossed'] = np.any(crossed)
        
        breakpoints.append(row)
    
    return pd.DataFrame(breakpoints)


def compute_method_stability(
    summary_df: pd.DataFrame,
    method_col: str = 'method',
    delta_col: str = 'delta',
    loss_col: str = 'loss',
    threshold: float = 1.0
) -> pd.DataFrame:
    """
    Compute stability metrics for each method.
    
    Metrics:
    - max_tolerable_delta: Maximum delta where loss <= threshold
    - tolerance_span: Range of deltas where tolerable
    - failure_rate: Proportion of scenarios where method fails
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary results
    method_col : str
        Column name for method
    delta_col : str
        Column name for delta
    loss_col : str
        Column name for loss
    threshold : float
        Loss threshold (default: 1.0)
    
    Returns
    -------
    pd.DataFrame
        Stability metrics per method
    """
    stability = []
    
    for method in summary_df[method_col].unique():
        method_data = summary_df[summary_df[method_col] == method]
        
        # Max tolerable delta
        tolerable = method_data[loss_col] <= threshold
        if np.any(tolerable):
            max_tolerable = method_data.loc[tolerable, delta_col].max()
        else:
            max_tolerable = 0.0
        
        # Tolerance span (range of deltas where tolerable)
        if np.any(tolerable):
            deltas_tolerable = method_data.loc[tolerable, delta_col].values
            if len(deltas_tolerable) > 0:
                span = deltas_tolerable.max() - deltas_tolerable.min()
            else:
                span = 0.0
        else:
            span = 0.0
        
        # Failure rate (proportion of scenarios where method fails)
        # A scenario fails if loss > threshold at delta=0 or no tolerable deltas
        scenario_failures = []
        for scenario, g in method_data.groupby(['q_regime', 'outcome_model', 'effect_type']):
            g_tolerable = g[loss_col] <= threshold
            if not np.any(g_tolerable):
                scenario_failures.append(1.0)
            else:
                scenario_failures.append(0.0)
        
        failure_rate = np.mean(scenario_failures) if scenario_failures else np.nan
        
        stability.append({
            'method': method,
            'max_tolerable_delta': max_tolerable,
            'tolerance_span': span,
            'failure_rate': failure_rate,
        })
    
    return pd.DataFrame(stability)