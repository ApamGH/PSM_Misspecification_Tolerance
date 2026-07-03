#!/usr/bin/env python
"""
Quick test script for the data generator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from misspecification_tolerance.config import get_pilot_config
from misspecification_tolerance.data_generation import DataGenerator


def main():
    print("Testing Data Generator")
    print("=" * 60)
    
    # Create configuration and generator
    config = get_pilot_config()
    rng = np.random.default_rng(42)
    generator = DataGenerator(config, rng)
    
    # Test dataset generation for each regime
    regimes = ["functional", "interaction", "combined", "omitted", "irrelevant"]
    
    for regime in regimes:
        print(f"\nRegime: {regime}")
        print("-" * 40)
        
        for delta in [0.0, 0.5, 1.0, 1.5, 2.0]:
            try:
                data = generator.generate_dataset(
                    delta=delta,
                    q_regime=regime,
                    outcome_model="linear",
                    effect_type="constant",
                    n=200,
                )
                
                n1 = int(np.sum(data["T"]))
                n0 = int(200 - n1)
                pi_emp = n1 / 200
                
                print(f"  delta={delta:4.1f}: n1={n1:3d}, n0={n0:3d}, "
                      f"pi={pi_emp:.3f}, ATT={data['att_sample']:.3f}, "
                      f"ATO={data['ato_sample']:.3f}")
                
            except Exception as e:
                print(f"  delta={delta:4.1f}: FAILED - {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()