#!/usr/bin/env python
"""
Main Execution Script
=====================

Run the misspecification-tolerance simulation.

Usage:
    python scripts/run_simulation.py [--config CONFIG] [--resume CHECKPOINT]
    [--output DIR] [--csv] [--no-parallel]

Examples:
    # Run pilot simulation
    python scripts/run_simulation.py
    
    # Run main simulation
    python scripts/run_simulation.py --config configs/main_config.json
    
    # Resume from checkpoint
    python scripts/run_simulation.py --resume outputs/checkpoint.pkl
    
    # Export to CSV
    python scripts/run_simulation.py --csv
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from misspecification_tolerance.config import (
    SimulationConfig, get_pilot_config, get_main_config
)
from misspecification_tolerance.simulation import SimulationRunner
from misspecification_tolerance.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Run misspecification-tolerance simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (JSON or YAML)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV in addition to pickle"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        if args.config.endswith('.json'):
            config = SimulationConfig.from_json(args.config)
        elif args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config = SimulationConfig.from_yaml(args.config)
        else:
            raise ValueError(f"Unsupported config file format: {args.config}")
    else:
        print("Using default pilot configuration.")
        config = get_pilot_config()
    
    # Override with command-line arguments
    if args.output:
        config.output_dir = args.output
    if args.no_parallel:
        config.n_jobs = 1
    if args.verbose:
        config.verbose = True
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set up logger
    log_file = os.path.join(config.output_dir, "simulation.log")
    logger = setup_logger(
        "Simulation",
        log_file,
        level=logging.DEBUG if config.verbose else logging.INFO
    )
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("Misspecification-Tolerance Simulation")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config.__dict__}")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Run simulation
    print(f"\nStarting simulation with {config.R} replications...")
    logger.info(f"Starting simulation with {config.R} replications")
    
    runner = SimulationRunner(config, logger=logger)
    results = runner.run(resume_from=args.resume)
    
    # Save results
    output_path = os.path.join(config.output_dir, "results.pkl")
    runner.save_results(output_path)
    logger.info(f"Results saved to {output_path}")
    
    if args.csv:
        runner.export_to_csv(config.output_dir)
        logger.info(f"Results exported to CSV in {config.output_dir}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total replications: {len(results)}")
    print(f"Methods: {config.methods}")
    print(f"Sample size: {config.n}")
    print(f"Results saved to: {config.output_dir}")
    
    # Print boundaries if available
    if runner.boundaries is not None:
        print("\nTolerance Boundaries:")
        print(runner.boundaries[['method', 'delta_star_conservative']].head(10))
    
    # Print ratios if available
    if runner.ratios is not None and not runner.ratios.empty:
        print("\nRelative PSM Tolerance Ratios:")
        print(runner.ratios[['comparison', 'relative_tolerance']].head(10))
    
    print("=" * 60)
    
    logger.info("Simulation complete")


if __name__ == "__main__":
    main()