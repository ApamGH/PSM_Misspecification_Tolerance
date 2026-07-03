"""
Script to fix imports after reorganizing the project structure.
"""

import os
import re

def fix_imports_in_file(filepath):
    """Fix imports in a Python file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix relative imports
    patterns = [
        (r'from \.config import', r'from ..config import'),
        (r'from \.data_generation import', r'from ..data_generation import'),
        (r'from \.diagnostics import', r'from ..diagnostics import'),
        (r'from \.utils import', r'from ..utils import'),
        (r'from \.simulation import', r'from ..simulation import'),
        (r'from \.analysis import', r'from ..analysis import'),
        (r'from \.visualization import', r'from ..visualization import'),
    ]
    
    for old, new in patterns:
        content = content.replace(old, new)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Fixed imports in: {filepath}")

# Fix all estimator files
estimator_files = [
    'misspecification_tolerance/estimators/base.py',
    'misspecification_tolerance/estimators/psm.py',
    'misspecification_tolerance/estimators/ipw.py',
    'misspecification_tolerance/estimators/ow.py',
    'misspecification_tolerance/estimators/aipw.py',
    'misspecification_tolerance/estimators/fm.py',
    'misspecification_tolerance/estimators/eb.py',
    'misspecification_tolerance/estimators/factory.py',
]

for file in estimator_files:
    if os.path.exists(file):
        fix_imports_in_file(file)

# Fix main module files
main_files = [
    'misspecification_tolerance/simulation.py',
    'misspecification_tolerance/diagnostics.py',
    'misspecification_tolerance/data_generation.py',
    'misspecification_tolerance/visualization.py',
    'misspecification_tolerance/analysis.py',
]

for file in main_files:
    if os.path.exists(file):
        fix_imports_in_file(file)