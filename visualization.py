"""
Visualization Module
====================

Plotting functions for simulation results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import warnings

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.2)


class SimulationPlotter:
    """
    Plotting utilities for simulation results.
    
    Creates publication-quality plots for:
    - Tolerance boundaries
    - Diagnostic loss curves
    - Coverage rates
    - Method comparisons
    - Component diagnostics
    """
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        plt.style.use(style)
        sns.set_palette("colorblind")
        self.colors = sns.color_palette("colorblind", n_colors=8)
        self.method_colors = {
            'PSM': self.colors[0],
            'FM': self.colors[1],
            'IPW': self.colors[2],
            'OW': self.colors[3],
            'EB': self.colors[4],
            'AIPW_correct': self.colors[5],
            'AIPW_restricted': self.colors[6],
        }
    
    def plot_tolerance_boundaries(
        self,
        boundaries_df: pd.DataFrame,
        group_cols: Optional[List[str]] = None,
        method_col: str = 'method',
        boundary_col: str = 'delta_star_conservative',
        scenario_col: str = 'scenario_id',
        save: bool = False,
        path: str = "outputs/plots/tolerance_boundaries.png",
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Plot tolerance boundaries by method and scenario.
        
        Parameters
        ----------
        boundaries_df : pd.DataFrame
            Boundaries dataframe
        group_cols : List[str], optional
            Columns to group by for faceting
        method_col : str
            Column name for method
        boundary_col : str
            Column name for boundary value
        scenario_col : str
            Column name for scenario identifier
        save : bool
            Whether to save the figure
        path : str
            Output path
        figsize : Tuple[int, int]
            Figure size
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        # Determine grouping for facets
        if group_cols is None:
            group_cols = ['q_regime', 'outcome_model', 'effect_type']
        
        # Create figure with subplots
        n_rows = len(boundaries_df['q_regime'].unique())
        n_cols = len(boundaries_df['outcome_model'].unique())
        
        if n_rows * n_cols > 1:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        else:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes]
        
        plot_idx = 0
        
        for q_regime in sorted(boundaries_df['q_regime'].unique()):
            for outcome_model in sorted(boundaries_df['outcome_model'].unique()):
                # Filter data
                mask = (boundaries_df['q_regime'] == q_regime) & \
                       (boundaries_df['outcome_model'] == outcome_model)
                data = boundaries_df[mask]
                
                if data.empty:
                    continue
                
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Pivot for grouped bar plot
                pivot = data.pivot_table(
                    index=['effect_type'],
                    columns=method_col,
                    values=boundary_col
                )
                
                # Plot
                pivot.plot(kind='bar', ax=ax, legend=False)
                
                # Customize
                ax.set_title(f"{q_regime}\n{outcome_model}")
                ax.set_ylabel("Tolerance Boundary (δ*)")
                ax.set_xlabel("Treatment Effect")
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linewidth=0.5)
                
                # Rotate x labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
        
        # Add legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), 
                   bbox_to_anchor=(0.5, 1.02))
        
        plt.tight_layout()
        
        if save:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_diagnostic_loss(
        self,
        summary_df: pd.DataFrame,
        method_col: str = 'method',
        delta_col: str = 'delta',
        loss_col: str = 'loss',
        group_cols: Optional[List[str]] = None,
        threshold: float = 1.0,
        save: bool = False,
        path: str = "outputs/plots/diagnostic_loss.png",
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Plot diagnostic loss curves by delta.
        
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
        group_cols : List[str], optional
            Columns to group by for faceting
        threshold : float
            Loss threshold (default: 1.0)
        save : bool
            Whether to save the figure
        path : str
            Output path
        figsize : Tuple[int, int]
            Figure size
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if group_cols is None:
            group_cols = ['q_regime', 'outcome_model', 'effect_type']
        
        # Get unique values for faceting
        q_values = summary_df['q_regime'].unique()
        n_rows = len(q_values)
        
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize)
        if n_rows == 1:
            axes = [axes]
        
        for idx, q_regime in enumerate(sorted(q_values)):
            ax = axes[idx]
            data = summary_df[summary_df['q_regime'] == q_regime]
            
            # Plot each method
            for method in data[method_col].unique():
                method_data = data[data[method_col] == method]
                if not method_data.empty:
                    # Group by delta and take mean
                    grouped = method_data.groupby(delta_col)[loss_col].mean()
                    ax.plot(
                        grouped.index,
                        grouped.values,
                        label=method,
                        marker='o',
                        markersize=4,
                        color=self.method_colors.get(method, None),
                        linewidth=2
                    )
            
            # Add threshold line
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      alpha=0.5, linewidth=2, label='Threshold')
            
            # Customize
            ax.set_title(f"q_regime: {q_regime}")
            ax.set_xlabel("Misspecification (δ)")
            ax.set_ylabel("Diagnostic Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', ncol=2)
        
        plt.tight_layout()
        
        if save:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_coverage(
        self,
        summary_df: pd.DataFrame,
        method_col: str = 'method',
        delta_col: str = 'delta',
        coverage_col: str = 'coverage',
        group_cols: Optional[List[str]] = None,
        nominal_coverage: float = 0.95,
        save: bool = False,
        path: str = "outputs/plots/coverage.png",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot coverage rates by method and delta.
        
        Parameters
        ----------
        summary_df : pd.DataFrame
            Summary results
        method_col : str
            Column name for method
        delta_col : str
            Column name for delta
        coverage_col : str
            Column name for coverage
        group_cols : List[str], optional
            Columns to group by for faceting
        nominal_coverage : float
            Nominal coverage level (default: 0.95)
        save : bool
            Whether to save the figure
        path : str
            Output path
        figsize : Tuple[int, int]
            Figure size
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if group_cols is None:
            group_cols = ['q_regime', 'outcome_model', 'effect_type']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each method
        for method in summary_df[method_col].unique():
            method_data = summary_df[summary_df[method_col] == method]
            if not method_data.empty:
                # Group by delta and take mean
                grouped = method_data.groupby(delta_col)[coverage_col].mean()
                ax.plot(
                    grouped.index,
                    grouped.values,
                    label=method,
                    marker='o',
                    markersize=5,
                    color=self.method_colors.get(method, None),
                    linewidth=2
                )
        
        # Add nominal coverage line
        ax.axhline(y=nominal_coverage, color='red', linestyle='--', 
                  alpha=0.5, linewidth=2, label='Nominal 95%')
        
        # Add confidence band for acceptable coverage
        lower_band = nominal_coverage - 0.025
        upper_band = nominal_coverage + 0.025
        ax.axhspan(lower_band, upper_band, alpha=0.1, color='green', 
                  label='Acceptable range')
        
        # Customize
        ax.set_xlabel("Misspecification (δ)")
        ax.set_ylabel("Coverage")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', ncol=2)
        ax.set_ylim(0.5, 1.0)
        
        plt.tight_layout()
        
        if save:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_component_breakdown(
        self,
        summary_df: pd.DataFrame,
        method_col: str = 'method',
        delta_col: str = 'delta',
        components: List[str] = None,
        group_cols: Optional[List[str]] = None,
        save: bool = False,
        path: str = "outputs/plots/component_breakdown.png",
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Plot breakdown of diagnostic components.
        
        Parameters
        ----------
        summary_df : pd.DataFrame
            Summary results
        method_col : str
            Column name for method
        delta_col : str
            Column name for delta
        components : List[str]
            Component column names
        group_cols : List[str], optional
            Columns to group by for faceting
        save : bool
            Whether to save the figure
        path : str
            Output path
        figsize : Tuple[int, int]
            Figure size
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if components is None:
            components = ['bias', 'rmse', 'coverage_error', 'fragility', 
                         'overlap_degradation', 'estimand_drift']
        
        if group_cols is None:
            group_cols = ['q_regime', 'effect_type']
        
        # Get unique methods
        methods = summary_df[method_col].unique()
        n_methods = len(methods)
        
        fig, axes = plt.subplots(n_methods, 1, figsize=figsize)
        if n_methods == 1:
            axes = [axes]
        
        for idx, method in enumerate(methods):
            ax = axes[idx]
            data = summary_df[summary_df[method_col] == method]
            
            if data.empty:
                continue
            
            # Group by delta and take mean of components
            grouped = data.groupby(delta_col)[components].mean()
            
            # Plot stacked area
            ax.stackplot(
                grouped.index,
                grouped[components].values.T,
                labels=components,
                alpha=0.7
            )
            
            # Customize
            ax.set_title(f"Method: {method}")
            ax.set_xlabel("Misspecification (δ)")
            ax.set_ylabel("Component Value")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', ncol=2)
        
        plt.tight_layout()
        
        if save:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_method_comparison(
        self,
        boundaries_df: pd.DataFrame,
        method_col: str = 'method',
        boundary_col: str = 'delta_star_conservative',
        group_cols: Optional[List[str]] = None,
        save: bool = False,
        path: str = "outputs/plots/method_comparison.png",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot method comparison with confidence intervals.
        
        Parameters
        ----------
        boundaries_df : pd.DataFrame
            Boundaries dataframe
        method_col : str
            Column name for method
        boundary_col : str
            Column name for boundary value
        group_cols : List[str], optional
            Columns to group by for faceting
        save : bool
            Whether to save the figure
        path : str
            Output path
        figsize : Tuple[int, int]
            Figure size
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if group_cols is None:
            group_cols = ['q_regime', 'outcome_model', 'effect_type']
        
        # Create a scenario identifier
        boundaries_df['scenario'] = boundaries_df[group_cols].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each method as a boxplot across scenarios
        data_to_plot = []
        labels = []
        
        for method in sorted(boundaries_df[method_col].unique()):
            method_data = boundaries_df[boundaries_df[method_col] == method]
            values = method_data[boundary_col].dropna()
            if len(values) > 0:
                data_to_plot.append(values.values)
                labels.append(method)
        
        # Create boxplot
        bp = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': 'black'}
        )
        
        # Color boxes
        for i, box in enumerate(bp['boxes']):
            method = labels[i]
            color = self.method_colors.get(method, self.colors[i % len(self.colors)])
            box.set_facecolor(color)
            box.set_alpha(0.7)
        
        # Customize
        ax.set_xlabel("Method")
        ax.set_ylabel("Tolerance Boundary (δ*)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        
        if save:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_heatmap(
        self,
        boundaries_df: pd.DataFrame,
        method_col: str = 'method',
        boundary_col: str = 'delta_star_conservative',
        pivot_cols: List[str] = None,
        save: bool = False,
        path: str = "outputs/plots/tolerance_heatmap.png",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Create heatmap of tolerance boundaries.
        
        Parameters
        ----------
        boundaries_df : pd.DataFrame
            Boundaries dataframe
        method_col : str
            Column name for method
        boundary_col : str
            Column name for boundary value
        pivot_cols : List[str], optional
            Columns to use for pivoting (method and scenario)
        save : bool
            Whether to save the figure
        path : str
            Output path
        figsize : Tuple[int, int]
            Figure size
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if pivot_cols is None:
            pivot_cols = ['q_regime', 'outcome_model', 'effect_type']
        
        # Create scenario identifier
        boundaries_df['scenario'] = boundaries_df[pivot_cols].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        
        # Pivot for heatmap
        pivot = boundaries_df.pivot_table(
            index='scenario',
            columns=method_col,
            values=boundary_col
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
        
        # Add labels
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Tolerance Boundary (δ*)')
        
        # Add values in cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.2f}',
                                 ha='center', va='center',
                                 color='black' if value < 1 else 'white',
                                 fontsize=8)
        
        ax.set_title('Tolerance Boundaries Heatmap')
        
        plt.tight_layout()
        
        if save:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def create_report(
        self,
        boundaries_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        ratios_df: pd.DataFrame,
        output_dir: str = "outputs/report"
    ) -> None:
        """
        Create a complete report with all plots.
        
        Parameters
        ----------
        boundaries_df : pd.DataFrame
            Boundaries dataframe
        summary_df : pd.DataFrame
            Summary results
        ratios_df : pd.DataFrame
            Relative ratios
        output_dir : str
            Output directory for report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Generating report...")
        
        # 1. Tolerance boundaries
        self.plot_tolerance_boundaries(
            boundaries_df,
            save=True,
            path=str(output_dir / "tolerance_boundaries.png")
        )
        
        # 2. Diagnostic loss curves
        self.plot_diagnostic_loss(
            summary_df,
            save=True,
            path=str(output_dir / "diagnostic_loss.png")
        )
        
        # 3. Coverage
        self.plot_coverage(
            summary_df,
            save=True,
            path=str(output_dir / "coverage.png")
        )
        
        # 4. Method comparison
        self.plot_method_comparison(
            boundaries_df,
            save=True,
            path=str(output_dir / "method_comparison.png")
        )
        
        # 5. Component breakdown
        self.plot_component_breakdown(
            summary_df,
            save=True,
            path=str(output_dir / "component_breakdown.png")
        )
        
        # 6. Heatmap
        self.plot_heatmap(
            boundaries_df,
            save=True,
            path=str(output_dir / "tolerance_heatmap.png")
        )
        
        # 7. Ratios plot
        if ratios_df is not None and not ratios_df.empty:
            self.plot_relative_ratios(
                ratios_df,
                save=True,
                path=str(output_dir / "relative_ratios.png")
            )
        
        print(f"Report saved to {output_dir}")
    
    def plot_relative_ratios(
        self,
        ratios_df: pd.DataFrame,
        comparison_col: str = 'comparison',
        ratio_col: str = 'relative_tolerance',
        save: bool = False,
        path: str = "outputs/plots/relative_ratios.png",
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot relative PSM tolerance ratios.
        
        Parameters
        ----------
        ratios_df : pd.DataFrame
            Relative ratios dataframe
        comparison_col : str
            Column name for comparison
        ratio_col : str
            Column name for ratio
        save : bool
            Whether to save the figure
        path : str
            Output path
        figsize : Tuple[int, int]
            Figure size
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract method names
        ratios_df['method'] = ratios_df[comparison_col].str.split(':').str[1]
        
        # Boxplot of ratios
        data_to_plot = []
        labels = []
        
        for method in sorted(ratios_df['method'].unique()):
            method_data = ratios_df[ratios_df['method'] == method]
            values = method_data[ratio_col].dropna()
            if len(values) > 0:
                data_to_plot.append(values.values)
                labels.append(method)
        
        bp = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': 'black'}
        )
        
        # Color boxes
        for i, box in enumerate(bp['boxes']):
            color = self.method_colors.get(labels[i], self.colors[i % len(self.colors)])
            box.set_facecolor(color)
            box.set_alpha(0.7)
        
        # Add reference line at 1 (equal tolerance)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal to PSM')
        
        # Add reference line at 0.5 and 2
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        ax.axhline(y=2.0, color='gray', linestyle=':', alpha=0.3)
        
        # Customize
        ax.set_xlabel("Method")
        ax.set_ylabel("Relative Tolerance (PSM / Method)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text annotation
        ax.text(0.02, 0.98, "R > 1: PSM more tolerant\nR = 1: Equal\nR < 1: PSM less tolerant",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig