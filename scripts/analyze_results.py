#!/usr/bin/env python
"""
Results Analysis Script
=======================

Analyze simulation results after they've been generated.

Usage:
    python scripts/analyze_results.py --input results.pkl [--output DIR] [--summary]
    [--compare-methods] [--plot]

Examples:
    # Load and summarize results
    python scripts/analyze_results.py --input outputs/results.pkl --summary
    
    # Compare methods
    python scripts/analyze_results.py --input outputs/results.pkl --compare-methods
    
    # Generate plots
    python scripts/analyze_results.py --input outputs/results.pkl --plot
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from misspecification_tolerance.analysis import (
    load_results,
    compute_method_ranks,
    compute_best_worst_methods,
    compute_summary_statistics,
    identify_breakpoints,
    compute_method_stability,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to results file (.pkl)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for analysis outputs"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics"
    )
    
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Compare methods"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    
    # Extract dataframes
    replication_df = pd.DataFrame(results.get('replication_results', []))
    summary_df = results.get('summary_results')
    boundaries_df = results.get('boundaries')
    ratios_df = results.get('ratios')
    
    print(f"Loaded {len(replication_df)} replications")
    if summary_df is not None:
        print(f"Loaded {len(summary_df)} summary rows")
    if boundaries_df is not None:
        print(f"Loaded {len(boundaries_df)} boundaries")
    
    # Set output directory
    output_dir = args.output or Path(args.input).parent / "analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Summary statistics
    if args.summary and summary_df is not None:
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        # Basic statistics per method
        stats = compute_summary_statistics(summary_df)
        print("\nPer-method summary:")
        print(stats.to_string(index=False))
        
        # Save to CSV
        stats.to_csv(Path(output_dir) / "method_summary.csv", index=False)
        print(f"\nSummary saved to {output_dir}/method_summary.csv")
    
    # Method comparison
    if args.compare_methods and boundaries_df is not None:
        print("\n" + "=" * 60)
        print("METHOD COMPARISON")
        print("=" * 60)
        
        # Method ranks
        ranked = compute_method_ranks(boundaries_df)
        print("\nMethod ranks (1 = best):")
        print(ranked[['method', 'rank']].drop_duplicates().sort_values('rank').to_string(index=False))
        
        # Best and worst methods
        best_worst = compute_best_worst_methods(boundaries_df)
        print("\nBest and worst methods:")
        print(best_worst[['best_method', 'worst_method']].value_counts().to_string())
        
        # Method stability
        if summary_df is not None:
            stability = compute_method_stability(summary_df)
            print("\nMethod stability:")
            print(stability.to_string(index=False))
            
            stability.to_csv(Path(output_dir) / "method_stability.csv", index=False)
        
        # Save ranks
        ranked.to_csv(Path(output_dir) / "method_ranks.csv", index=False)
        print(f"\nResults saved to {output_dir}/method_ranks.csv")
    
    # Generate plots
    if args.plot:
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        
        try:
            from misspecification_tolerance.visualization import SimulationPlotter
            plotter = SimulationPlotter()
            
            # Plot tolerance boundaries
            if boundaries_df is not None:
                plotter.plot_tolerance_boundaries(
                    boundaries_df,
                    save=True,
                    path=Path(output_dir) / "tolerance_boundaries.png"
                )
                print(f"  Tolerance boundaries plot saved to {output_dir}/tolerance_boundaries.png")
            
            # Plot diagnostic loss
            if summary_df is not None:
                plotter.plot_diagnostic_loss(
                    summary_df,
                    save=True,
                    path=Path(output_dir) / "diagnostic_loss.png"
                )
                print(f"  Diagnostic loss plot saved to {output_dir}/diagnostic_loss.png")
                
                # Plot coverage
                plotter.plot_coverage(
                    summary_df,
                    save=True,
                    path=Path(output_dir) / "coverage.png"
                )
                print(f"  Coverage plot saved to {output_dir}/coverage.png")
            
        except ImportError as e:
            print(f"  Could not generate plots: {e}")
            print("  Make sure matplotlib and seaborn are installed.")
    
    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()