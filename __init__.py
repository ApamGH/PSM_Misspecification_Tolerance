"""
Misspecification Tolerance Framework for Propensity Score Methods
================================================================

A diagnostic boundary framework for observational causal inference.

This package implements the simulation design described in:
"Design Fragility and Misspecification Tolerance in PSM: 
A Diagnostic Boundary Framework for Observational Causal Inference"

Key Features:
- 6 covariates with equicorrelation structure
- 5 misspecification regimes (functional, interaction, combined, omitted, irrelevant)
- 7 estimation methods (PSM, FM, IPW, OW, EB, AIPW_correct, AIPW_restricted)
- Comprehensive diagnostic loss framework
- Tolerance boundary estimation
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import SimulationConfig
from .simulation import SimulationRunner

__all__ = [
    "SimulationConfig",
    "SimulationRunner",
]