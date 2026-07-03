# Misspecification-Tolerance Framework for Propensity Score Methods

A diagnostic boundary framework for evaluating the robustness of propensity score methods under treatment-model misspecification.

## Overview

This package implements the simulation framework described in:

> "Design Fragility and Misspecification Tolerance in PSM: A Diagnostic Boundary Framework for Observational Causal Inference"

The framework treats misspecification as a graded design stressor and defines a tolerance boundary for each method: the maximum misspecification level at which the method's combined diagnostic loss remains within acceptable limits.

## Key Features

- **7 Estimation Methods**: PSM, FM, IPW, OW, EB, AIPW_correct, AIPW_restricted
- **5 Misspecification Regimes**: functional, interaction, combined, omitted, irrelevant
- **6 Diagnostic Components**: bias, RMSE, coverage, fragility, overlap, drift
- **Parallel Processing**: Multi-core support for fast simulations
- **Checkpointing**: Resume interrupted simulations
- **Comprehensive Visualization**: Publication-quality plots

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/misspecification-tolerance.git
cd misspecification-tolerance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -e .
