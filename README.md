# Misspecification-Tolerance Framework for Propensity Score Methods

A diagnostic boundary framework for evaluating the robustness of propensity score methods under treatment-model misspecification.

## Overview

This repository contains the simulation code, estimator implementations, diagnostic routines, analysis scripts, and manuscript-supporting outputs for the methodological study:

> **Design Fragility and Misspecification Tolerance in PSM: A Diagnostic Boundary Framework for Observational Causal Inference**

The framework treats misspecification as a graded design stressor rather than a simple correct-versus-incorrect condition. It defines a **misspecification-tolerance boundary** for each method as the maximum level of treatment-model misspecification at which the method’s combined diagnostic loss remains within acceptable limits.

The project is designed to support reproducible simulation-based evaluation of propensity-score and balancing-based causal designs under different forms of model misspecification. It is intended for researchers, statisticians, causal inference practitioners, and applied data analysts interested in robustness diagnostics, overlap quality, weighting behaviour, and design fragility in observational studies.

## Key Features

- **Seven estimation methods**
  - Propensity Score Matching
  - Full Matching
  - Inverse Probability Weighting
  - Overlap Weighting
  - Entropy Balancing
  - Correctly specified Augmented Inverse Probability Weighting
  - Restricted Augmented Inverse Probability Weighting

- **Five misspecification regimes**
  - Functional-form misspecification
  - Interaction misspecification
  - Combined misspecification
  - Omitted-confounder misspecification
  - Irrelevant-variable misspecification

- **Six diagnostic components**
  - Bias
  - RMSE
  - Coverage failure
  - Fragility
  - Overlap degradation
  - Estimand drift

- **Additional functionality**
  - Configurable simulation settings
  - Parallel processing for large simulation runs
  - Checkpointing for interrupted simulations
  - Manuscript-ready tables and figures
  - Reusable analysis and visualisation routines

## Repository Structure

```text
misspecification-tolerance/
├── misspecification_tolerance/
│   ├── __init__.py
│   ├── analysis.py
│   ├── config.py
│   ├── data_generation.py
│   ├── diagnostics.py
│   ├── simulation.py
│   ├── utils.py
│   ├── visualization.py
│   └── estimators/
│       ├── __init__.py
│       ├── base.py
│       ├── psm.py
│       ├── fm.py
│       ├── ipw.py
│       ├── ow.py
│       ├── eb.py
│       ├── aipw.py
│       └── factory.py
├── configs/
│   ├── pilot_config.json
│   ├── main_config.json
│   └── core_config.json
├── scripts/
│   ├── run_simulation.py
│   └── analyze_results.py
├── tests/
│   ├── test_data_generation.py
│   └── test_diagnostics.py
├── outputs/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/misspecification-tolerance.git
cd misspecification-tolerance
```

Replace the URL above with the actual GitHub repository URL.

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If the repository includes a valid `pyproject.toml`, `setup.py`, or `setup.cfg`, the package may also be installed in editable mode:

```bash
pip install -e .
```

## Basic Usage

The repository is designed around configuration files stored in the `configs/` directory. These configuration files define the simulation design, including sample size, number of replications, estimation methods, misspecification regimes, diagnostic components, and output locations.

## Running the Pilot Simulation

A pilot simulation is useful for checking whether the installation, estimators, diagnostics, and output-writing routines are working correctly.

```bash
python scripts/run_simulation.py --config configs/pilot_config.json
```

The pilot simulation should be run before the full simulation. It is expected to run faster and produce smaller output files.

## Running the Main Simulation

To run the full simulation:

```bash
python scripts/run_simulation.py --config configs/main_config.json
```

Depending on the number of replications, methods, regimes, and available processing cores, the main simulation may take considerable time.

## Analysing Simulation Results

After running the simulation, analyse the output files using:

```bash
python scripts/analyze_results.py --input outputs/main
```

The analysis routine is intended to generate summary results, diagnostic comparisons, tolerance-boundary estimates, relative tolerance ratios, and manuscript-supporting tables and figures.

## Output Files

Simulation and analysis outputs are stored in the `outputs/` directory. Depending on the run configuration, this directory may contain:

```text
outputs/
├── pilot/
├── main/
├── checkpoints/
├── figures/
├── tables/
├── logs/
└── manuscript_analysis/
```

Typical output files include:

- `summary_results.csv`
- `tolerance_boundaries.csv`
- `relative_tolerance_ratios.csv`
- `simulation.log`
- manuscript-ready tables
- manuscript-ready figures

The existing outputs are provided to support transparency and reproducibility. Users may delete or regenerate these files by rerunning the simulation and analysis scripts.

## Running Tests

To run the test suite:

```bash
pytest
```

The tests check selected components of the data-generation and diagnostic routines. Users are encouraged to run the tests after installing dependencies and before running the full simulation.

## Methodological Summary

The framework evaluates how different propensity-score and balancing-based estimators respond to increasing treatment-model misspecification. Instead of asking only whether a model is correctly specified, the framework asks how much misspecification a realised design can tolerate before its diagnostic performance becomes unacceptable.

For each method and misspecification regime, the simulation tracks diagnostic loss across several components, including estimation error, uncertainty performance, overlap quality, design fragility, and estimand drift. These components are combined into a diagnostic loss measure, from which a method-specific tolerance boundary is estimated.

The resulting tolerance boundary provides a design-level robustness summary: methods with higher tolerance boundaries remain diagnostically acceptable under more severe misspecification, whereas methods with lower boundaries become fragile earlier.

## Estimation Methods

The simulation compares the following methods:

| Abbreviation | Method |
|---|---|
| PSM | Propensity Score Matching |
| FM | Full Matching |
| IPW | Inverse Probability Weighting |
| OW | Overlap Weighting |
| EB | Entropy Balancing |
| AIPW_correct | Correctly specified Augmented Inverse Probability Weighting |
| AIPW_restricted | Restricted Augmented Inverse Probability Weighting |

## Misspecification Regimes

The simulation evaluates five misspecification regimes:

| Regime | Description |
|---|---|
| Functional | Treatment model omits or distorts functional-form terms |
| Interaction | Treatment model omits relevant interaction terms |
| Combined | Treatment model combines multiple forms of misspecification |
| Omitted | Treatment model omits relevant confounding variables |
| Irrelevant | Treatment model includes variables that do not induce prognostic imbalance |

## Diagnostic Components

The diagnostic loss framework includes six components:

| Component | Description |
|---|---|
| Bias | Average deviation between estimated and target treatment effects |
| RMSE | Root mean squared estimation error |
| Coverage | Failure of confidence intervals to achieve nominal coverage |
| Fragility | Instability of design or estimator performance under stress |
| Overlap | Degradation of common-support or weighting quality |
| Drift | Movement away from the intended estimand or target population |

## Reproducibility Notes

For reproducibility, users should record:

- Python version
- Operating system
- Package versions
- Simulation configuration file
- Random seed
- Number of replications
- Number of processing cores
- Date and time of execution

The `requirements.txt` file provides the main package dependencies. For strict reproducibility, users may generate a full environment record using:

```bash
pip freeze > environment_snapshot.txt
```

## Recommended Workflow for Contributors

Before making changes, pull the latest version:

```bash
git pull
```

After editing files, check the repository status:

```bash
git status
```

Add changed files:

```bash
git add .
```

Commit the changes with a clear message:

```bash
git commit -m "Describe the change made"
```

Push the changes to GitHub:

```bash
git push
```

A typical workflow is therefore:

```bash
git pull
git status
git add .
git commit -m "Update simulation diagnostics"
git push
```

## Troubleshooting

### `pip install -e .` fails

This usually means the repository does not yet contain a valid packaging file such as `pyproject.toml`, `setup.py`, or `setup.cfg`.

Use:

```bash
pip install -r requirements.txt
```

### Python cannot find the package

Make sure you are running commands from the root folder of the repository. You may also need to install the package in editable mode if a packaging file is available:

```bash
pip install -e .
```

### Tests fail after editing code

Run:

```bash
pytest
```

Check the failing test name and inspect the relevant module. Diagnostic and simulation functions should be tested after any structural change.

### Simulation takes too long

Reduce the number of replications in the configuration file or run the pilot configuration first. Large simulation runs may require multiple CPU cores and substantial processing time.

### Output files are not generated

Check that the output directory exists and that the configuration file specifies a valid output path. Also inspect the log file for error messages.

## Manuscript Citation
Note: The associated manuscript is not published at the moment. The correct citation will be placed here when it is finally published. If you want to collaborate as well on this, please get in touch with me at bapam@bolgatu.edu.gh.

<!-- If you use or adapt this repository, please cite the associated manuscript:

```text
Apam, B. (2026). Design fragility and misspecification tolerance in PSM: 
A diagnostic boundary framework for observational causal inference.
``` -->

## Software Citation

If citing the repository directly, use:

```text
Apam, B. (2026). Misspecification-tolerance framework for propensity score methods 
[Computer software]. GitHub. https://github.com/yourusername/misspecification-tolerance
```

Replace the GitHub URL with the actual repository link.

## Licence

This project is released under the MIT Licence. See the `LICENSE` file for details.

## Author

**Dr. Benjamin Apam**  
Lecturer | Statistician | Data Analyst  
Bolgatanga Technical University  

## Contact

For questions, collaboration, or manuscript-related enquiries, please contact the repository author through GitHub or the institutional contact details provided in the associated manuscript.

## Disclaimer

This repository is provided for research and reproducibility purposes. The simulation results depend on the specified data-generating process, misspecification regimes, diagnostic thresholds, and estimator implementations. Users should adapt the framework carefully when applying it to new causal inference settings.